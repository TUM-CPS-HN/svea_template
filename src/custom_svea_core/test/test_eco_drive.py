"""Unit and property tests for the eco-drive stack (no ROS needed).

Run from the repository root:

    python3 -m pytest src/custom_svea_core/test/test_eco_drive.py -q
"""

import math
import random
import sys
from pathlib import Path

import numpy as np
import pytest

try:
    from custom_svea_core.eco_drive import (EcoDrivePlanner, Gate, GuardLight, LightTiming, PlannerConfig,
                                            RedLightGuard, Route, RouteProgress, green_windows,
                                            phase_at_offset, random_layout)
    from custom_svea_core.eco_drive.sim import LightSpec, SimLight, run_episode
except Exception:  # no ROS in this environment: import the package directly
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "custom_svea_core"))
    from eco_drive import (EcoDrivePlanner, Gate, GuardLight, LightTiming, PlannerConfig,  # type: ignore
                           RedLightGuard, Route, RouteProgress, green_windows,
                           phase_at_offset, random_layout)
    from eco_drive.sim import LightSpec, SimLight, run_episode  # type: ignore


# ----------------------------------------------------------------- green windows

@pytest.mark.parametrize("state,left,first", [
    ("Rd", 3.0, (5.0, 10.0)),   # red 3 s, yellow 2 s, then green 5 s
    ("YG", 1.0, (1.0, 6.0)),
    ("Gr", 2.0, (0.0, 2.0)),    # currently green
    ("YR", 1.0, (8.0, 13.0)),   # 1 + red 5 + yellow 2
])
def test_nominal_windows(state, left, first):
    w = green_windows(state, left, LightTiming())
    assert (w[0].start, w[0].end) == pytest.approx(first)
    # consecutive windows are one period apart
    assert w[1].start - w[0].start == pytest.approx(LightTiming().period) or state == "Gr"


def test_jitter_windows_are_inside_nominal_ones():
    nom = green_windows("Rd", 2.0, LightTiming(alpha=0.0))
    rob = green_windows("Rd", 2.0, LightTiming(alpha=0.5))
    assert len(rob) >= 1
    for r in rob:
        n = next(n for n in nom if n.index == r.index)
        assert n.start <= r.start and r.end <= n.end
        assert r.duration < n.duration


@pytest.mark.parametrize("alpha", [0.0, 0.3, 0.8])
def test_guaranteed_windows_hold_for_every_realisation(alpha):
    """Property test: whatever the jitter does, the light is green inside the windows."""
    timing = LightTiming(alpha=alpha)
    rng = random.Random(42)
    for trial in range(200):
        state = rng.choice(["Rd", "YG", "Gr", "YR"])
        left = rng.uniform(0.01, timing.nominal(state))
        light = SimLight(timing, state, left, random.Random(trial))
        for w in green_windows(state, left, timing, horizon=60):
            for t in np.linspace(w.start + 1e-6, w.end - 1e-6, 7):
                light.advance(t)
                assert light.state_at(t) == "Gr", (state, left, w, t)


# ------------------------------------------------------------------------ route

def test_route_straight_roundtrip():
    r = Route([[0, 0], [10, 0]])
    assert r.length == pytest.approx(10.0, abs=1e-6)
    s, lat = r.project(4.0, 0.3)
    assert s == pytest.approx(4.0, abs=0.06) and lat == pytest.approx(0.3, abs=1e-3)
    x, y = r.xy(7.5)
    assert (float(x), float(y)) == pytest.approx((7.5, 0.0), abs=1e-6)


def test_route_fillet_curvature_and_lap_unrolling():
    sq = [[0, 0], [10, 0], [10, 10], [0, 10]]
    r = Route(sq, closed=True, fillet_radius=2.0)
    # corners rounded with R = 2 m -> curvature 0.5 somewhere, 0 on straights
    assert np.max(np.abs(r.kappa)) == pytest.approx(0.5, rel=0.1)
    assert abs(float(r.curvature(5.0))) < 1e-3
    prog = RouteProgress(r)
    s_prev = -1.0
    path = np.arange(0.0, 2.5 * r.length, 0.2)  # drive 2.5 laps
    for s in path:
        x, y = r.xy(s)
        s_new = prog.update(float(x), float(y))
        assert s_new > s_prev
        s_prev = s_new
    assert s_prev == pytest.approx(path[-1], abs=0.06)


# ---------------------------------------------------------------------- planner

def _check_plan(plan, s_light, window, cfg, v_max=1.2):
    # the hard guarantee is time_margin_min; the rest of time_margin is soft
    m = cfg.time_margin_min
    t_cross = plan.time_at(s_light)
    lo = window.start + m if window.start > 0 else 0.0
    assert lo - 1e-3 <= t_cross <= window.end - m + 1e-3
    t_exit = plan.time_at(s_light + cfg.clear_dist)
    assert t_exit <= window.end - m + 1e-3
    # dense check: the car is strictly before the light until the crossing time
    tt = np.linspace(0, plan.duration, 2000)
    s, v, _ = plan.sample(tt)
    assert np.all(s[tt < t_cross - 1e-3] < s_light + 1e-6)
    assert np.all(np.diff(s) >= -1e-9), "s(t) must be monotone"
    assert np.all(v <= v_max + 1e-4) and np.all(v >= -1e-6)


def test_plan_crosses_inside_window_and_respects_limits():
    tm = LightTiming()
    p = EcoDrivePlanner()
    for state, left in [("Rd", 3.0), ("Gr", 4.5), ("YR", 1.0), ("YG", 0.5)]:
        gates = [Gate("a", 8.0, green_windows(state, left, tm))]
        plan = p.plan(0.0, 0.0, gates, s_goal=15.0)
        assert plan.status == "goal"
        _check_plan(plan, 8.0, plan.crossings["a"], p.cfg)
        assert plan.a.max() <= p.limits.a_max + 1e-6 and plan.a.min() >= p.limits.a_min - 1e-6
        assert plan.s[-1] == pytest.approx(15.0, abs=1e-4) and plan.v[-1] == pytest.approx(0.0, abs=1e-4)


def test_unreachable_window_is_skipped():
    # green for only 1 more second but the light is 8 m away: must take the next green
    p = EcoDrivePlanner()
    plan = p.plan(0.0, 0.0, [Gate("a", 8.0, green_windows("Gr", 1.0, LightTiming()))], s_goal=15.0)
    assert plan.crossings["a"].index == 1


def test_hold_when_light_is_unknown():
    p = EcoDrivePlanner()
    plan = p.plan(0.0, 1.0, [Gate("a", 6.0, [])], s_goal=15.0)
    assert plan.status == "hold"
    assert plan.s[-1] <= 6.0 - p.cfg.stop_margin + 1e-4
    assert plan.v[-1] == pytest.approx(0.0, abs=1e-4)


def test_multi_light_plan():
    tm = LightTiming()
    gates = [Gate("a", 6.0, green_windows("Rd", 2.0, tm)), Gate("b", 12.0, green_windows("Gr", 1.0, tm))]
    p = EcoDrivePlanner()
    plan = p.plan(0.0, 0.5, gates, s_goal=18.0)
    assert plan.ok and set(plan.crossings) == {"a", "b"}
    for g in gates:
        _check_plan(plan, g.s, plan.crossings[g.name], p.cfg)


# ----------------------------------------------------------------- guard + loop

def test_guard_brakes_for_red_and_ignores_green():
    guard = RedLightGuard()
    red = GuardLight("a", 5.0, green_windows("Rd", 4.0, LightTiming()))
    green = GuardLight("a", 5.0, green_windows("Gr", 4.0, LightTiming()))
    v, who = guard.limit(3.0, 1.0, 1.0, [red])  # 2 m away: no need to brake yet
    assert who is None and v == 1.0
    v, who = guard.limit(4.6, 1.0, 1.0, [red])  # 0.4 m away at 1 m/s: brake
    assert who == "a" and v < 0.7
    v, who = guard.limit(4.6, 1.0, 1.0, [green])
    assert who is None and v == 1.0
    v, who = guard.limit(4.95, 0.0, 0.5, [red])  # between the margin line and the light
    assert v == 0.0


@pytest.mark.parametrize("seed", range(6))
def test_closed_loop_never_runs_a_red_light(seed):
    rng = random.Random(seed)
    tm = LightTiming(alpha=rng.choice([0.0, 0.5]))
    st = rng.choice(["Rd", "YG", "Gr", "YR"])
    lights = [LightSpec("a", rng.uniform(4, 10), tm, st, rng.uniform(0.1, tm.nominal(st)))]
    ep = run_episode(lights, 14.0, EcoDrivePlanner(config=PlannerConfig()), seed=seed, speed_noise=0.03)
    assert not ep.violations
    assert not math.isnan(ep.arrival_time)
    assert ep.crossings["a"][1] == "Gr"


def test_laggy_actuator_still_makes_the_first_green():
    """Regression: sim_svea.py reaches only 1/1.27 of the commanded speed and drops the
    command once a second. Replanning from the measured state then slid the plan past the
    green window; replanning from the reference + integral action must keep it."""
    tm = LightTiming()
    lights = [LightSpec("a", 9.46, tm, "Rd", 4.6)]
    ep = run_episode(lights, 17.68, EcoDrivePlanner(), speed_noise=0.05,
                     actuator_gain=1 / 1.27, actuator_dropout=True)
    assert not ep.violations
    assert ep.crossings["a"][0] < 11.6  # first green window is [6.6, 11.6]


# ----------------------------------------------------------------- load tests: close + many lights

def test_lights_closer_than_the_car_are_passed_together():
    """Light b is inside the clearing zone of light a (0.4 m apart): both must be green at
    once. Phase end points interleave and must be merged, not left out of order."""
    tm = LightTiming()
    lights = [LightSpec("a", 6.0, tm, "Rd", 2.0), LightSpec("b", 6.4, tm, "Rd", 2.5)]
    ep = run_episode(lights, 12.0, EcoDrivePlanner(), seed=0)
    assert not ep.violations
    assert not math.isnan(ep.arrival_time)
    assert all(p.status != "failed" for _, p in ep.plans)


def test_never_holds_between_chained_lights():
    """a is green now, b (0.5 m later) only much later: the car may not pass a and then
    wait between them (that is inside the intersection). It must hold before a."""
    planner = EcoDrivePlanner()
    gates = [Gate("a", 3.0, green_windows("Gr", 4.0, LightTiming())),
             Gate("b", 3.5, green_windows("Rd", 4.0, LightTiming()))]
    plan = planner.plan(0.0, 0.0, gates, s_goal=10.0)
    s_wait = plan.s[np.argmax(plan.v[1:] < 1e-3) + 1] if np.any(plan.v[1:] < 1e-3) else None
    if s_wait is not None and s_wait < 9.9:
        assert s_wait <= 3.0 - planner.cfg.stop_margin + 0.05
    for name, w in plan.crossings.items():  # whatever it chose, both crossings are inside green
        g = next(g for g in gates if g.name == name)
        assert w.start - 1e-6 <= plan.time_at(g.s) <= w.end + 1e-6


def test_guard_answers_a_red_chain_light_at_the_first_stop_line():
    guard = RedLightGuard()
    tm = LightTiming()
    a = GuardLight("a", 5.0, green_windows("Gr", 4.0, tm))
    b = GuardLight("b", 5.4, [])  # red / unknown, inside a's clearing zone
    v, who = guard.limit(4.5, 1.0, 1.0, [a, b])
    assert who == "b"
    # braking target is a's stop line (5.0), not b's (5.4): never stop between them
    assert v == pytest.approx(math.sqrt(2 * guard.decel * (5.0 - guard.line_margin - 4.5 - guard.latency)))


def test_random_layout_has_close_clusters_and_valid_offsets():
    tm = LightTiming()
    lay = random_layout(57.0, 32, seed=3, timing=tm, closed=True)
    s = np.array([p.s for p in lay])
    assert len(lay) == 32 and np.all(np.diff(s) > 0)
    assert np.min(np.diff(s)) < 0.9  # some lights closer than the car can stop between
    assert s.min() >= 2.0 and s.max() <= 57.0 - 2.0
    for p in lay:
        st, left = phase_at_offset(p.offset, tm)
        assert 0 < left <= tm.nominal(st) + 1e-9


@pytest.mark.parametrize("n,seed", [(16, 0), (32, 1)])
def test_dense_random_lap_is_safe_and_finishes(n, seed):
    tm = LightTiming(alpha=0.5)
    lay = random_layout(57.0, n, seed=seed, timing=tm, closed=True)
    lights = [LightSpec(p.name, p.s, tm, *phase_at_offset(p.offset, tm)) for p in lay]
    ep = run_episode(lights, 57.0, EcoDrivePlanner(), seed=seed, t_max=300.0, speed_noise=0.03,
                     actuator_gain=1 / 1.27, actuator_dropout=True)
    assert not ep.violations
    assert not math.isnan(ep.arrival_time)
    assert all(ep.crossings[L.name][1] == "Gr" for L in lights)


def test_impassable_chain_waits_safely():
    """Two lights 0.4 m apart whose greens never overlap: nobody can pass, the car must wait
    before the first one forever and never cross on red."""
    red, green, yellow = 5.0, 5.0, 2.0
    tm = LightTiming(red, yellow, green)
    lights = [LightSpec("a", 5.0, tm, "Rd", 1.0), LightSpec("b", 5.4, tm, "Gr", 0.5)]  # 7 s apart
    ep = run_episode(lights, 10.0, EcoDrivePlanner(), seed=0, t_max=40.0)
    assert not ep.violations
    assert "a" not in ep.crossings and ep.s.max() < 5.0

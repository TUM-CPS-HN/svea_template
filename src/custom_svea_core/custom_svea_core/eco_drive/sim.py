"""Offline closed-loop simulator for planner, tracker and lights (no ROS needed).

It mirrors the ROS setup closely enough to tune and stress-test the stack on
a laptop:

* ``SimLight`` uses the same state machine and jitter as ``traffic_light.py``.
* ``SimCar1D`` uses the longitudinal model of ``sim_svea.py``: a first-order ESC
  with tau = 0.1 s, acceleration clipped at 2 m/s², optional speed noise,
  integrated at 40 Hz.
* ``run_episode`` runs the same loop as the ROS node: control at 10 Hz,
  replanning every ``replan_period`` seconds, and the red-light guard on every
  command. It returns a full log and checks for red-light violations.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass, field

import numpy as np

from .energy import battery_energy
from .planner import EcoDrivePlanner, Gate, Plan
from .tracker import GuardLight, RedLightGuard, SpeedTracker, replan_origin
from .traffic_signal import TRANSITIONS, LightTiming, green_windows

__all__ = ["SimLight", "SimCar1D", "LightSpec", "Episode", "run_episode"]


class SimLight:
    """Replica of the ROS traffic light state machine (``traffic_light.py``)."""

    def __init__(self, timing: LightTiming, state: str = "Rd", time_left: float | None = None,
                 rng: random.Random | None = None):
        self.timing = timing
        self.rng = rng or random.Random(0)
        self.state = state
        self.t_end = self._duration(state) if time_left is None else time_left
        self.history = [(0.0, state)]

    def _duration(self, state):
        d = self.timing.nominal(state)
        if state in ("Rd", "Gr"):
            d += self.rng.uniform(-self.timing.alpha, self.timing.alpha)
        return d

    def advance(self, t: float):
        while t >= self.t_end:
            self.state = TRANSITIONS[self.state]
            self.history.append((self.t_end, self.state))
            self.t_end += self._duration(self.state)

    def observe(self, t: float):
        self.advance(t)
        return self.state, self.t_end - t

    def state_at(self, t: float) -> str:
        """State at an earlier time t (from the recorded history)."""
        st = self.history[0][1]
        for t0, s in self.history:
            if t0 <= t:
                st = s
            else:
                break
        return st


class SimCar1D:
    TAU = 0.1
    ACCEL_MAX = 2.0

    def __init__(self, s=0.0, v=0.0, speed_noise=0.0, rng=None, gain=1.0, dropout=False):
        self.s, self.v = s, v
        self.speed_noise = speed_noise
        self.rng = rng or np.random.default_rng(0)
        self.gain = gain  # steady-state speed gain of the actuator (sim_svea.py: 1/1.27)
        self.dropout = dropout  # sim_svea.py zeroes the command just after every full second
        self.t = 0.0

    def step(self, v_cmd: float, dt: float):
        self.t += dt
        v_cmd = self.gain * v_cmd
        if self.dropout and (self.t % 1.0) < 0.05:
            v_cmd = 0.0
        if self.speed_noise and v_cmd > 0.0:  # ESC fluctuations; a zero command holds still
            v_cmd = v_cmd + self.rng.normal(0, self.speed_noise)
        a = np.clip((v_cmd - self.v) / self.TAU, -self.ACCEL_MAX, self.ACCEL_MAX)
        s_old = self.s
        self.s += self.v * dt + 0.5 * a * dt * dt
        self.v = max(0.0, self.v + a * dt)
        return s_old, a


@dataclass
class LightSpec:
    name: str
    s: float
    timing: LightTiming = field(default_factory=LightTiming)
    state: str = "Rd"
    time_left: float | None = None


@dataclass
class Episode:
    t: np.ndarray
    s: np.ndarray
    v: np.ndarray
    a: np.ndarray
    v_cmd: np.ndarray
    plans: list  # (t_plan, Plan)
    crossings: dict  # name -> (t, state)
    violations: list  # (name, t, state)
    guard_interventions: int
    arrival_time: float
    lights: dict  # name -> SimLight (with history)
    solve_times: list

    @property
    def acc_cost(self) -> float:
        return float(np.sum(self.a[:-1] ** 2 * np.diff(self.t)))

    @property
    def battery_energy(self) -> float:
        return battery_energy(self.t, self.v, self.a[:-1])

    @property
    def max_abs_accel(self) -> float:
        return float(np.max(np.abs(self.a)))


def run_episode(
    lights: list[LightSpec],
    s_goal: float,
    planner: EcoDrivePlanner | None = None,
    mode: str = "eco",  # 'eco' | 'naive'
    v0: float = 0.0,
    s0: float = 0.0,
    t_max: float = 120.0,
    control_dt: float = 0.1,
    sim_dt: float = 0.025,
    replan_period: float = 0.5,
    naive_speed: float = 1.2,
    speed_noise: float = 0.0,
    seed: int = 0,
    use_guard: bool = True,
    planner_alpha: float | None = None,
    actuator_gain: float = 1.0,
    actuator_dropout: bool = False,
    replan_from_reference: bool = True,
) -> Episode:
    """Simulate one drive from ``s0`` to ``s_goal``.

    ``mode='naive'`` is the baseline the tutorial starts from: drive at a
    constant ``naive_speed`` and let the red-light guard brake at the line
    when the light is red (the guard plays the role of a careful human).

    ``planner_alpha`` lets the planner assume a different jitter than the true
    one (for robustness experiments); by default it knows the real alpha.
    """
    rng = random.Random(seed)
    sim_lights = {L.name: SimLight(L.timing, L.state, L.time_left, rng) for L in lights}
    car = SimCar1D(s0, v0, speed_noise, np.random.default_rng(seed), actuator_gain, actuator_dropout)
    tracker = SpeedTracker(v_max=planner.limits.v_max if planner else 1.2)
    guard = RedLightGuard(clear_dist=planner.cfg.clear_dist if planner else 0.6,
                          v_max=planner.limits.v_max if planner else 1.2)

    log_t, log_s, log_v, log_a, log_cmd = [], [], [], [], []
    plans, solve_times = [], []
    crossings, violations = {}, []
    interventions = 0
    plan: Plan | None = None
    t_plan = -math.inf
    v_cmd, a_cmd = 0.0, 0.0
    t, arrival = 0.0, math.nan
    steps_per_ctrl = int(round(control_dt / sim_dt))
    step = 0

    def observations(now):
        obs = {}
        for L in lights:
            st, tl = sim_lights[L.name].observe(now)
            timing = L.timing if planner_alpha is None else LightTiming(
                L.timing.red, L.timing.yellow, L.timing.green, planner_alpha)
            obs[L.name] = green_windows(st, tl, timing, horizon=120.0)
        return obs

    while t < t_max:
        if step % steps_per_ctrl == 0:
            obs = observations(t)
            if mode == "eco":
                if plan is None or t - t_plan >= replan_period - 1e-9:
                    gates = [Gate(L.name, L.s, obs[L.name]) for L in lights]
                    if replan_from_reference and plan is not None:
                        s_start, v_start, _ = replan_origin(plan, t - t_plan, car.s, car.v,
                                                            barriers=[L.s for L in lights])
                    else:
                        s_start, v_start = car.s, car.v
                    new = planner.plan(s_start, v_start, gates, s_goal=s_goal, a_prev=a_cmd)
                    solve_times.append(new.solve_time)
                    plans.append((t, new))
                    # a failed replan does not throw away a plan we are still following
                    # (the guard keeps us safe if that plan's crossing becomes unsafe)
                    if new.ok or plan is None or not plan.ok or t - t_plan >= plan.duration:
                        plan, t_plan = new, t
                v_cmd, _, _, a_cmd = tracker.command(plan, t - t_plan, car.s, car.v, control_dt)
            else:
                d_goal = s_goal - car.s
                v_cmd = min(naive_speed, math.sqrt(2 * 1.0 * max(d_goal, 0.0)))
            if use_guard:
                g_lights = [GuardLight(L.name, L.s, obs[L.name]) for L in lights]
                v_safe, who = guard.limit(car.s, car.v, v_cmd, g_lights)
                if who is not None and mode == "eco":
                    interventions += 1
                v_cmd = v_safe

        s_old, a = car.step(v_cmd, sim_dt)
        t += sim_dt
        step += 1
        for L in lights:
            if s_old < L.s <= car.s and L.name not in crossings:
                sim_lights[L.name].advance(t)
                st = sim_lights[L.name].state
                crossings[L.name] = (t, st)
                if st != "Gr":
                    violations.append((L.name, t, st))
        log_t.append(t), log_s.append(car.s), log_v.append(car.v), log_a.append(a), log_cmd.append(v_cmd)
        if car.s >= s_goal - 0.05 and car.v < 0.05 and math.isnan(arrival):
            arrival = t
            break

    return Episode(np.array(log_t), np.array(log_s), np.array(log_v), np.array(log_a),
                   np.array(log_cmd), plans, crossings, violations, interventions,
                   arrival, sim_lights, solve_times)

"""Tracking the planned speed profile, plus an independent red-light guard.

``SpeedTracker`` turns the time-parametrised plan into a velocity command:

    v_cmd = v_plan(t + τ_esc)  +  k_s · (s_plan(t) - s_meas)

The first term is feed-forward. It previews the plan by the ESC time constant so
the first-order speed loop of the car lands on the planned speed on time. The
second term is position feedback, so the car does not drift ahead of or behind
the plan, which matters because the plan crosses the light at a precise time.

``RedLightGuard`` is a safety layer that does not trust the optimizer. Every
control tick it checks each light ahead:

    "If I keep going at the speed I am about to command, will I be between the
     light and the clearing point only while the light is surely green?"

If the answer is no, it caps the command with a braking envelope,
``v ≤ sqrt(2 b d)``, which stops the car before the line. A plan that is
followed well never triggers the guard. It only acts when something unexpected
happens: a stale light topic, a solver failure, poor tracking, or timing
outside the jitter model. Each intervention is counted and published.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .planner import Plan
from .traffic_signal import GreenWindow

__all__ = ["SpeedTracker", "RedLightGuard", "GuardLight", "replan_origin"]


def replan_origin(plan: Plan | None, t: float, s_meas: float, v_meas: float,
                  max_ds: float = 0.4, max_dv: float = 0.3, barriers=()):
    """State to start the next plan from: ``(s0, v0, from_reference)``.

    Replanning from the *measured* state every 0.5 s quietly forgives every bit
    of tracking lag: the plan slides later and later, and it can miss a green
    window it could easily have made. While the car is close to the plan we
    therefore replan from the *reference* state (where the plan says we should
    be). The tracker's feedback then removes the lag. Only after a real
    deviation (or without a plan) do we restart from the measurement.

    ``barriers`` are the arc lengths of the lights. If a light lies between the
    car and the reference, the next plan would start on the wrong side of it
    (e.g. the reference crept past a red light while the car waited, and the
    plan would no longer see that light's window), so we use the measurement.
    """
    if plan is not None and plan.ok:
        s_r, v_r, _ = plan.sample(t)
        s_r, v_r = float(s_r), float(v_r)
        lo, hi = min(s_r, s_meas), max(s_r, s_meas)
        straddles = any(lo < b <= hi + 0.02 for b in barriers)
        if abs(s_r - s_meas) <= max_ds and abs(v_r - v_meas) <= max_dv and not straddles:
            return s_r, v_r, True
    return s_meas, v_meas, False


@dataclass
class SpeedTracker:
    k_s: float = 0.8  # position feedback gain [1/s]
    k_i: float = 0.6  # integral gain on the speed error [1/s]
    preview: float = 0.1  # ESC time constant / actuation delay [s]
    v_max: float = 1.2
    max_lead: float = 0.3  # cap on the position feedback correction [m/s]
    max_integral: float = 0.45  # cap on the integral correction [m/s]
    creep_tol: float = 0.03  # stopped further than this short of a standstill plan: creep [m]

    def __post_init__(self):
        self.integral = 0.0

    def command(self, plan: Plan, t: float, s_meas: float, v_meas: float | None = None,
                dt: float = 0.1) -> tuple[float, float, float, float]:
        """Return ``(v_cmd, s_plan, v_plan, a_plan)`` at time ``t`` since the plan started.

        The integral term learns a steady gain/offset error of the speed loop
        (e.g. an ESC that only reaches 80 % of the commanded speed). The command
        may therefore exceed ``v_max`` by up to ``max_integral``, so that the
        *measured* speed can reach the limit.
        """
        s_p, v_p, a_p = (float(x) for x in plan.sample(t))
        _, v_ff, _ = plan.sample(t + self.preview)
        v_ff = float(v_ff)
        if v_ff < 1e-3 and v_p < 1e-3:
            # plan says "stand still": hold, and do not let the integrator creep us forward.
            # If the car stopped short of where the plan stands (actuator lag), creep up to it.
            self.integral = min(self.integral, 0.0)
            gap = s_p - s_meas
            v_creep = min(self.k_s * gap + 0.05, self.max_lead) if gap > self.creep_tol else 0.0
            return v_creep, s_p, v_p, a_p
        if v_meas is not None:
            self.integral += self.k_i * (v_p - v_meas) * dt
            self.integral = max(-self.max_integral, min(self.max_integral, self.integral))
        corr = max(-self.max_lead, min(self.max_lead, self.k_s * (s_p - s_meas)))
        v_cmd = v_ff + corr + self.integral
        return max(0.0, min(self.v_max + self.max_integral, v_cmd)), s_p, v_p, a_p


@dataclass
class GuardLight:
    name: str
    s: float  # arc length of the light (unrolled)
    windows: list[GreenWindow]  # guaranteed green windows relative to now (empty = unknown/red)


@dataclass
class RedLightGuard:
    decel: float = 1.0  # braking used by the envelope [m/s^2]
    line_margin: float = 0.10  # aim to stop this far before the light [m]
    latency: float = 0.15  # ESC lag + control period: distance covered before braking acts [s]
    clear_dist: float = 0.60  # same clearing distance as the planner
    lookahead: float = 6.0  # only check lights closer than this [m]
    tol: float = 0.05  # timing tolerance [s]
    stop_margin: float = 0.30  # planner stop margin: lights closer than clear + this form a chain
    accel: float = 0.5  # plausible pull-away acceleration for the exit-time check [m/s^2]
    accel_max: float = 2.0  # physical limit of the ESC: bounds the earliest possible entry [m/s^2]
    v_max: float = 1.2

    def limit(self, s: float, v_meas: float, v_cmd: float, lights: list[GuardLight]):
        """Cap ``v_cmd`` if continuing would cross a light outside a green window.

        Returns:
            ``(v_safe, intervening_light_name_or_None)``
        """
        v_safe, who = v_cmd, None
        lights = sorted(lights, key=lambda L: L.s)
        # close lights form a chain: the car cannot stop between them, so a red light
        # in a chain must be answered at the stop line of the chain's first light
        anchors = []
        for i, L in enumerate(lights):
            chained = i and L.s - self.stop_margin < lights[i - 1].s + self.clear_dist
            anchors.append(anchors[-1] if chained else L.s)
        for L, anchor in zip(lights, anchors):
            if s >= L.s:
                continue  # front already past the light: keep going, never stop inside
            target = anchor if s < anchor else L.s  # inside a chain already: stop at this light
            d_line = target - self.line_margin - s
            if d_line > self.lookahead:
                continue
            if self._crossing_ok(L, s, max(v_meas, v_cmd)):
                continue
            # braking envelope, shortened by the distance covered during the actuation lag;
            # between the margin line and the light itself the cap is zero (hold)
            d_eff = max(d_line - v_meas * self.latency, 0.0)
            cap = math.sqrt(2.0 * self.decel * d_eff)
            if cap < v_safe:
                v_safe, who = cap, L.name
        return v_safe, who

    def _time_to(self, d: float, v0: float, a: float | None = None) -> float:
        """Time to cover ``d`` from speed ``v0``, accelerating at ``a`` up to ``v_max``."""
        a = self.accel if a is None else a
        vm = max(self.v_max, v0)
        d_acc = (vm**2 - v0**2) / (2 * a)
        if d <= d_acc:
            return (-v0 + math.sqrt(v0**2 + 2 * a * d)) / a
        return (vm - v0) / a + (d - d_acc) / vm

    def _crossing_ok(self, L: GuardLight, s: float, v: float) -> bool:
        """Is the crossing safe for every way the car may actually continue?

        * Entry: even at full ESC acceleration the car must not reach the light before
          green starts (so a car that accelerates harder than expected cannot sneak in).
        * Exit: a plausible continuation (keep the speed, or pull away at a modest
          ``accel``) must have the rear clear before green ends. Constant speed alone
          would wrongly reject every start from rest (at 0.1 m/s the car "needs" 14 s
          to clear a 1.4 m chain).
        """
        if v < 1e-3:
            return True  # standing still never crosses
        d_in, d_out = L.s - s, L.s + self.clear_dist - s
        t_in_early = self._time_to(d_in, v, self.accel_max)
        t_out = min(d_out / v, self._time_to(d_out, v))
        return any(w.start <= t_in_early and t_out <= w.end + self.tol for w in L.windows)

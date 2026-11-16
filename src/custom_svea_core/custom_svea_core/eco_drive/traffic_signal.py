"""Traffic-light timing model and *guaranteed* green windows.

The light node (``custom_svea_examples/scripts/traffic_light.py``) cycles

    Rd --(RED_TIME ± alpha)--> YG --(YELLOW_TIME)--> Gr --(GREEN_TIME ± alpha)--> YR --(YELLOW_TIME)--> Rd

and publishes the current state plus the exact time left in that state. The
duration of every *future* red/green phase is only known up to ``± alpha``
(it is drawn uniformly at random when the phase starts), so a planner that
assumes nominal durations can be wrong by up to ``k * alpha`` seconds after
``k`` cycles.

This module propagates an interval ``[earliest, latest]`` for every future phase
switch and returns, for each upcoming green phase, the interval during which the
light is green **for every possible realisation of the jitter**:

    guaranteed green = [latest possible start, earliest possible end]

Crossing inside that interval is safe no matter how the dice fall. With
``alpha = 0`` this collapses to the nominal schedule.

Pure Python, no ROS imports, so it can be unit tested anywhere.
"""

from __future__ import annotations

from dataclasses import dataclass

__all__ = ["LightTiming", "GreenWindow", "green_windows", "state_at", "TRANSITIONS", "SAFE_STATES"]

TRANSITIONS = {"Rd": "YG", "YG": "Gr", "Gr": "YR", "YR": "Rd"}
SAFE_STATES = ("Gr",)  # the only state in which the stop line may be crossed


@dataclass(frozen=True)
class LightTiming:
    """Phase durations of a light; must match the parameters of the light node."""

    red: float = 5.0
    yellow: float = 2.0
    green: float = 5.0
    alpha: float = 0.0  # uniform jitter on red and green durations [s]

    @property
    def period(self) -> float:
        return self.red + self.green + 2.0 * self.yellow

    def nominal(self, state: str) -> float:
        return {"Rd": self.red, "Gr": self.green}.get(state, self.yellow)

    def jitter(self, state: str) -> float:
        return self.alpha if state in ("Rd", "Gr") else 0.0


@dataclass(frozen=True)
class GreenWindow:
    """A time interval (relative to 'now') in which the light is surely green."""

    start: float
    end: float
    index: int  # 0 = the green phase that is current or next, 1 = the one after, ...

    @property
    def duration(self) -> float:
        return self.end - self.start

    def contains(self, t: float, tol: float = 0.0) -> bool:
        return self.start - tol <= t <= self.end + tol


def green_windows(
    state: str,
    time_left: float,
    timing: LightTiming,
    horizon: float = 90.0,
    max_windows: int = 8,
) -> list[GreenWindow]:
    """Guaranteed-green intervals over ``[0, horizon]`` seconds from now.

    Args:
        state: current light state, one of ``'Rd', 'YG', 'Gr', 'YR'``.
        time_left: seconds remaining in the current state (exact, as published).
        timing: phase durations and jitter of this light.
        horizon: stop looking for windows beyond this time.
        max_windows: maximum number of windows returned.

    Returns:
        Chronologically sorted list of :class:`GreenWindow`. Windows that shrink
        to nothing because of accumulated jitter are dropped (the index still
        counts them, so ``index`` always refers to the physical green phase).
    """
    if state not in TRANSITIONS:
        raise ValueError(f"Unknown light state {state!r}")

    t_left = max(0.0, float(time_left))
    windows: list[GreenWindow] = []
    green_idx = 0

    # The current phase is fully known: it ends exactly in `t_left` seconds.
    if state == "Gr":
        if t_left > 0.0:
            windows.append(GreenWindow(0.0, t_left, green_idx))
        green_idx += 1

    end_lo = end_hi = t_left  # interval in which the current phase ends
    cur = state
    while end_lo <= horizon and len(windows) < max_windows:
        cur = TRANSITIONS[cur]
        start_lo, start_hi = end_lo, end_hi
        d, j = timing.nominal(cur), timing.jitter(cur)
        end_lo, end_hi = start_lo + d - j, start_hi + d + j
        if cur == "Gr":
            if start_hi < end_lo and start_hi <= horizon:
                windows.append(GreenWindow(start_hi, end_lo, green_idx))
            green_idx += 1
    return windows


def state_at(state: str, time_left: float, timing: LightTiming, t: float) -> str:
    """Nominal (alpha ignored) light state ``t`` seconds from now. Used for plotting."""
    cur, remaining = state, max(0.0, time_left)
    while t >= remaining:
        t -= remaining
        cur = TRANSITIONS[cur]
        remaining = timing.nominal(cur)
    return cur

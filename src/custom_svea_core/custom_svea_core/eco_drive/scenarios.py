"""Random traffic-light layouts for load tests.

``random_layout`` scatters lights along a route. Some of them come in clusters
of 2-4 lights only 0.4-1.0 m apart, closer than the 0.9 m the car needs to stop
between two lights (stop margin + clearing distance). Each light also gets a
random phase offset, so the lights are not synchronised.

Lights in a cluster are *coordinated* by default (a green wave: each one turns
green about gap / 1 m/s later than the one before, ±0.75 s). Without that, two
lights whose clearing zones overlap may never be green together, and then no
planner can pass them. ``coordinate_clusters=False`` builds such adversarial
layouts on purpose; the car must then wait safely, forever if need be.

Used by ``eco_drive_analysis.py loadtest`` (offline) and by
``eco_drive.launch.py random_lights:=N`` (ROS).
"""

from __future__ import annotations

import random
from dataclasses import dataclass

from .traffic_signal import TRANSITIONS, LightTiming

__all__ = ["LightPlacement", "random_layout", "phase_at_offset"]


@dataclass
class LightPlacement:
    name: str
    s: float  # arc length along the route [m]
    offset: float  # time into the cycle (from the start of red) when the lights start [s]


def phase_at_offset(offset: float, timing: LightTiming) -> tuple[str, float]:
    """``(state, time_left)`` of a light that started red ``offset`` seconds ago (nominal timing)."""
    state, remaining = "Rd", timing.red
    offset = offset % timing.period
    while offset >= remaining:
        offset -= remaining
        state = TRANSITIONS[state]
        remaining = timing.nominal(state)
    return state, remaining - offset


def random_layout(
    length: float,
    n: int,
    seed: int = 0,
    timing: LightTiming = LightTiming(),
    cluster_prob: float = 0.35,
    cluster_size: tuple[int, int] = (2, 4),
    close_gap: tuple[float, float] = (0.4, 1.0),
    min_gap: float = 1.2,
    start_margin: float = 2.0,
    end_margin: float = 2.5,
    closed: bool = False,
    coordinate_clusters: bool = True,
    wave_speed: float = 1.0,
    wave_jitter: float = 0.75,
) -> list[LightPlacement]:
    """Place ``n`` lights on ``[start_margin, length - end_margin]``.

    Single lights keep at least ``min_gap`` to every other light. A cluster puts
    its lights ``close_gap`` apart. On a closed route the start margin also
    applies across the lap wrap. If the route is too short for ``n`` lights the
    layout holds as many as fit.
    """
    rng = random.Random(seed)
    lo, hi = start_margin, length - (start_margin if closed else end_margin)
    placed: list[float] = []

    def free(group):
        return all(lo <= x <= hi for x in group) and all(
            abs(x - y) >= min_gap for x in group for y in placed)

    offsets: dict[float, float] = {}
    tries = 0
    while len(placed) < n and tries < 5000:
        tries += 1
        k = 1
        if rng.random() < cluster_prob:
            k = min(rng.randint(*cluster_size), n - len(placed))
        x = rng.uniform(lo, hi)
        group = [x]
        for _ in range(k - 1):
            group.append(group[-1] + rng.uniform(*close_gap))
        if free(group):
            placed += group
            off = rng.uniform(0.0, timing.period)
            for i, g in enumerate(group):
                if i and coordinate_clusters:
                    # downstream light turns green later by about the travel time
                    off = off - (g - group[i - 1]) / wave_speed + rng.uniform(-wave_jitter, wave_jitter)
                elif i:
                    off = rng.uniform(0.0, timing.period)
                offsets[g] = off % timing.period
    placed.sort()
    return [LightPlacement(f"light_{i:02d}", x, offsets[x]) for i, x in enumerate(placed)]

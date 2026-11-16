"""Run the planner in a separate process so IPOPT never blocks the control loop.

The ROS node submits ``(s0, v0, gates, s_goal, a_prev)`` to a one-worker
``ProcessPoolExecutor`` and keeps tracking the previous plan at 10 Hz until
the new one comes back. The worker builds its own planner and route once, in
``init_worker``.
"""

from __future__ import annotations

from .planner import EcoDrivePlanner, PlannerConfig, VehicleLimits, Weights
from .route import Route

_planner: EcoDrivePlanner | None = None


def init_worker(limits: dict, weights: dict, config: dict, route_points, route_closed: bool,
                fillet_radius: float, use_curvature_limit: bool):
    global _planner
    lim = VehicleLimits(**limits)
    speed_fn = None
    if use_curvature_limit:
        route = Route(route_points, closed=route_closed, fillet_radius=fillet_radius)
        speed_fn = lambda s: route.speed_limit(s, lim.v_max, lim.a_lat_max)  # noqa: E731
    _planner = EcoDrivePlanner(lim, Weights(**weights), PlannerConfig(**config), speed_fn)
    # build and JIT-warm the most common NLP structures before the car moves
    _planner.plan(0.0, 0.0, [], s_goal=5.0)


def solve(s0, v0, gates, s_goal, a_prev):
    assert _planner is not None, "init_worker was not called"
    return _planner.plan(s0, v0, gates, s_goal=s_goal, a_prev=a_prev)

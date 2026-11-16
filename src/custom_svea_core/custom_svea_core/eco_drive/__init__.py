"""Eco-driving through signalised intersections (Tutorial 2).

Pure Python + numpy + CasADi. Nothing in this package imports ROS, so it can be
used, tested and plotted on any laptop. The ROS glue lives in
``custom_svea_examples/scripts/eco_drive.py``.

Modules:
    traffic_signal: light timing model and guaranteed green windows
    route:          arc-length parametrised, filleted route (x, y) <-> s
    planner:        multi-phase free-time CasADi/IPOPT speed planner
    tracker:        plan tracking law and independent red-light guard
    energy:         energy metrics (sum of a^2 dt and battery energy model)
    scenarios:      random light layouts for load tests
    sim:            offline closed-loop simulator used by tests and analysis
"""

from .traffic_signal import LightTiming, GreenWindow, green_windows, state_at
from .route import Route, RouteProgress
from .energy import accel_cost, battery_energy
from .planner import VehicleLimits, Weights, PlannerConfig, Gate, Plan, EcoDrivePlanner
from .tracker import SpeedTracker, RedLightGuard, GuardLight, replan_origin
from .scenarios import LightPlacement, random_layout, phase_at_offset

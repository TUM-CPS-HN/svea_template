#! /usr/bin/env python3
r"""Eco-drive: drive through traffic lights on green, fast, with gentle acceleration.

This node builds on the Tutorial 1 pure pursuit node. Steering is still done
by ``PurePursuitController`` and the lidar safety lock is kept. The constant
``target_velocity`` is replaced by an optimal, time-parametrised speed plan.

Pipeline (one control tick every 0.1 s)::

    odometry ──► RouteProgress ──► s, v ─┐
    /light_x/state, /light_x/time_left ──► green_windows() ─┐
                                          │                  │
                       every 0.5 s ──► EcoDrivePlanner (CasADi/IPOPT, separate process)
                                          │
                                          ▼
                     SpeedTracker: v_plan(t+τ) + k (s_plan − s)
                                          │
                     RedLightGuard (independent braking envelope)
                                          │
                     lidar safety lock (Tutorial 1)
                                          │
              steering = pure pursuit on the filleted route ──► actuation

All of the maths lives in ``custom_svea_core.eco_drive`` and is pure Python,
so it is unit-tested and analysed offline. See
``docs/tutorials/2_eco_drive_solution.md``.

Run (simulation)::

    ros2 launch custom_svea_examples eco_drive.launch.py
    ros2 launch custom_svea_examples eco_drive.launch.py lap:=true      # bonus: 4 lights, full lap

Published (relative to the node, e.g. ``/svea_a/eco_drive/...``):
    viz (MarkerArray)        plan coloured by speed, ghost car, stop lines, labels
    st_plan, st_actual (Path)  space-time diagram (pose.x = t, pose.y = s) for Foxglove plots
    telemetry/* (Float32)    v_plan, v_cmd, v_meas, a_plan, s, s_plan, energy_J, acc_cost, solve_ms, guard
    status (String)          human readable status line
"""

import csv
import math
import os
import time
from concurrent.futures import ProcessPoolExecutor
from multiprocessing import get_context

import numpy as np
from geometry_msgs.msg import Point, PoseStamped
from nav_msgs.msg import Path
from sensor_msgs.msg import LaserScan
from std_msgs.msg import ColorRGBA, Empty, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from custom_svea_core import rosonic as rx
from custom_svea_core.controllers.pure_pursuit import PurePursuitController
from custom_svea_core.eco_drive import (Gate, GuardLight, LightTiming, PlannerConfig, RedLightGuard, Route,
                                        RouteProgress, SpeedTracker, VehicleLimits, Weights, green_windows,
                                        replan_origin)
from custom_svea_core.eco_drive import worker
from custom_svea_core.eco_drive.energy import CarParams
from custom_svea_core.interfaces import ActuationInterface, LocalizationInterface, ShowPath

LIGHT_RGBA = {"Rd": (0.85, 0.15, 0.15), "YG": (0.95, 0.75, 0.0), "Gr": (0.15, 0.75, 0.25),
              "YR": (0.95, 0.75, 0.0), "??": (0.5, 0.5, 0.5)}


def speed_rgba(v, v_max):
    """Blue (slow) -> green -> yellow (fast), like the analysis plots."""
    x = max(0.0, min(1.0, v / max(v_max, 1e-6)))
    stops = [(0.0, (0.27, 0.0, 0.33)), (0.35, (0.2, 0.4, 0.55)), (0.65, (0.15, 0.7, 0.45)), (1.0, (0.99, 0.9, 0.15))]
    for (x0, c0), (x1, c1) in zip(stops, stops[1:]):
        if x <= x1:
            f = (x - x0) / (x1 - x0)
            return tuple(c0[i] + f * (c1[i] - c0[i]) for i in range(3))
    return stops[-1][1]


class LightMonitor:
    """Latest state/time_left of one light, with stale and not-running detection."""

    STALE_AFTER = 0.6  # [s] without messages -> unknown (treated as red)

    def __init__(self, name, s_local, timing):
        self.name, self.s_local, self.timing = name, s_local, timing
        self.state, self.time_left = "??", 0.0
        self.t_state = self.t_left = -math.inf
        self._frozen_since = None
        self._last_left = None

    def on_state(self, msg):
        self.state, self.t_state = msg.data, time.monotonic()

    def on_time_left(self, msg):
        now = time.monotonic()
        tl = float(msg.data)
        # the light node publishes a constant time_left while it waits for its start trigger
        if self._last_left is not None and abs(tl - self._last_left) < 1e-4:
            self._frozen_since = self._frozen_since or now
        else:
            self._frozen_since = None
        self._last_left = tl
        self.time_left, self.t_left = tl, now

    @property
    def running(self):
        return self._frozen_since is None or time.monotonic() - self._frozen_since < 0.5

    def windows(self, now):
        """Guaranteed green windows relative to ``now`` (empty if unknown/stale/not running)."""
        if self.state not in ("Rd", "YG", "Gr", "YR"):
            return []
        if now - min(self.t_state, self.t_left) > self.STALE_AFTER or not self.running:
            return []
        tl = self.time_left - (now - self.t_left)
        # state and time_left arrive as two messages; reject an inconsistent pair at a transition
        if tl > self.timing.nominal(self.state) + self.timing.jitter(self.state) + 0.25:
            return []
        return green_windows(self.state, max(tl, 0.0), self.timing, horizon=90.0)


class eco_drive(rx.Node):

    DELTA_TIME = 0.1  # control period [s]

    ## Parameters

    is_sim = rx.Parameter(True)
    points = rx.Parameter([-0.38, 0.96, 15.0, -2.69])  # route waypoints [x0, y0, x1, y1, ...]
    closed = rx.Parameter(False)  # True: drive laps around the waypoints
    prepend_start = rx.Parameter(True)  # start the (open) route at the car's position
    fillet_radius = rx.Parameter(1.0)

    lights = rx.Parameter(["light_a"])  # names of traffic light nodes
    light_positions = rx.Parameter([7.0, -0.79])  # [x, y] per light (same as the light nodes)
    red_time = rx.Parameter(5.0)  # must match traffic_light.py
    yellow_time = rx.Parameter(2.0)
    green_time = rx.Parameter(5.0)
    alpha = rx.Parameter(0.0)  # light timing jitter (same as the light nodes)
    start_lights = rx.Parameter(True)  # trigger lights started with wait_for_start:=true

    v_max = rx.Parameter(1.2)
    a_max = rx.Parameter(1.0)
    a_min = rx.Parameter(-0.8)
    a_lat_max = rx.Parameter(0.8)
    w_time = rx.Parameter(1.0)
    w_acc = rx.Parameter(4.0)
    w_jerk = rx.Parameter(0.5)
    horizon = rx.Parameter(25.0)
    replan_period = rx.Parameter(0.5)
    use_curvature_limit = rx.Parameter(True)

    use_lidar_safety = rx.Parameter(True)
    log_dir = rx.Parameter("~/eco_drive_logs")

    ## Interfaces

    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    path = ShowPath()

    ## Startup

    def on_startup(self):
        self.controller = PurePursuitController()
        self.limits = VehicleLimits(self.v_max, self.a_max, self.a_min, self.a_lat_max)
        self.weights = Weights(time=self.w_time, acc=self.w_acc, jerk=self.w_jerk)
        self.cfg = PlannerConfig(horizon=self.horizon)
        self.timing = LightTiming(self.red_time, self.yellow_time, self.green_time, self.alpha)
        self.tracker = SpeedTracker(v_max=self.v_max)
        self.guard = RedLightGuard(clear_dist=self.cfg.clear_dist, stop_margin=self.cfg.stop_margin,
                                   v_max=self.v_max)

        self.route = None  # built on the first valid odometry message
        self.plan, self.t_plan = None, None
        self.future = None
        self.t_last_submit = -math.inf
        self.a_cmd = 0.0
        self.finished = False
        self.t0 = time.monotonic()
        self.energy_J, self.acc_cost = 0.0, 0.0
        self.v_prev, self.a_meas = 0.0, 0.0
        self.guard_ticks, self.ticks = 0, 0
        self.st_actual = []

        # light topics
        names = list(self.lights)
        pos = np.asarray(self.light_positions, float).reshape(-1, 2)
        assert len(pos) == len(names), "light_positions must hold one [x, y] pair per light"
        self.monitors = []
        self._light_xy = pos
        for name in names:
            mon = LightMonitor(name, None, self.timing)
            self.create_subscription(String, f"/{name}/state", mon.on_state, 10)
            self.create_subscription(Float32, f"/{name}/time_left", mon.on_time_left, 10)
            self.monitors.append(mon)
        self.start_pub = self.create_publisher(Empty, "/planned_trajectory/start", 1)

        # outputs
        self.viz_pub = self.create_publisher(MarkerArray, "~/viz", 1)
        self.st_plan_pub = self.create_publisher(Path, "~/st_plan", 1)
        self.st_actual_pub = self.create_publisher(Path, "~/st_actual", 1)
        self.status_pub = self.create_publisher(String, "~/status", 1)
        self.telemetry = {k: self.create_publisher(Float32, f"~/telemetry/{k}", 10) for k in
                          ("v_plan", "v_cmd", "v_meas", "a_plan", "s", "s_plan", "energy_J",
                           "acc_cost", "solve_ms", "guard")}

        self.pool = None  # planner process, created once the route is known
        self.t_start_pub = -math.inf

        # Tutorial 1 lidar safety lock
        self.safety_lock = False
        self.k_th, self.k_v, self.c, self.points_limit = 0.15, 0.35, 0.2, 5
        if self.use_lidar_safety:
            self.create_subscription(LaserScan, f'/{"svea_a" if self.is_sim else "self"}/scan',
                                     self.lidar_callback, 1)

        self.actuation.enable_difflock()
        self._open_log()
        self.create_timer(self.DELTA_TIME, self.loop)
        self.create_timer(0.2, self.publish_viz)
        self.get_logger().info("Eco-drive node started, waiting for odometry ...")

    def _init_route(self, x, y):
        pts = np.asarray(self.points, float).reshape(-1, 2)
        if self.prepend_start and not self.closed:
            pts = np.vstack([[x, y], pts])
        self.route = Route(pts, closed=self.closed, fillet_radius=self.fillet_radius)
        self.progress = RouteProgress(self.route)
        self.s = self.progress.update(x, y)
        self.s_goal = None if self.closed else self.route.length
        for mon, (lx, ly) in zip(self.monitors, self._light_xy):
            mon.s_local, lat = self.route.project(lx, ly)
            if abs(lat) > 1.5:
                self.get_logger().warn(f"{mon.name} is {lat:.2f} m from the route; projecting it onto the path")
            self.get_logger().info(f"{mon.name}: s = {mon.s_local:.2f} m along the route")
        self.path.publish_path(self.route.x[::5], self.route.y[::5])
        self._route_pts = pts.tolist()
        self._start_pool()
        self.get_logger().info(f"Route ready: {self.route.length:.1f} m, {'lap' if self.closed else 'open'}")

    def _start_pool(self):
        """Planner process (spawn, because forking a process that runs ROS threads is unsafe)."""
        if self.pool is not None:
            self.pool.shutdown(wait=False, cancel_futures=True)
        self.future = None
        self.pool = ProcessPoolExecutor(
            max_workers=1, mp_context=get_context("spawn"), initializer=worker.init_worker,
            initargs=(vars(self.limits), vars(self.weights), vars(self.cfg), self._route_pts, self.closed,
                      self.fillet_radius, self.use_curvature_limit))

    ## Helpers

    def gates(self, s_now, now):
        """Lights ahead in unrolled arc length, with their guaranteed green windows."""
        out = []
        for mon in self.monitors:
            s_l = mon.s_local
            if self.route.closed:
                L = self.route.length
                s_l = s_l + L * math.ceil((s_now - self.cfg.clear_dist - s_l) / L)
            elif s_l + self.cfg.clear_dist < s_now:
                continue
            out.append((mon, s_l, mon.windows(now)))
        return out

    def _open_log(self):
        self.log_writer = None
        if not self.log_dir:
            return
        d = os.path.expanduser(self.log_dir)
        os.makedirs(d, exist_ok=True)
        fn = os.path.join(d, time.strftime("eco_drive_%Y%m%d_%H%M%S.csv"))
        self._log_file = open(fn, "w", newline="")
        cols = ["t", "x", "y", "s", "s_plan", "v", "v_plan", "v_cmd", "a_plan", "guard", "lidar_lock", "solve_ms"]
        for name in self.lights:
            cols += [f"{name}_state", f"{name}_s"]
        self.log_writer = csv.DictWriter(self._log_file, fieldnames=cols)
        self.log_writer.writeheader()
        self.get_logger().info(f"Logging to {fn}  (plot: eco_drive_analysis.py log {fn})")

    ## Main loop

    def loop(self):
        x, y, yaw, vel = self.localizer.get_state()
        if self.route is None:
            if x == 0.0 and y == 0.0 and vel == 0.0:
                return  # no odometry yet
            self._init_route(x, y)

        now = time.monotonic()
        s = self.progress.update(x, y)
        v = max(0.0, float(vel))
        self.ticks += 1
        gates = self.gates(s, now)

        # 1) planning (asynchronous)
        self._collect_plan()
        if self.future is None and now - self.t_last_submit >= self.replan_period and not self.finished:
            g = [Gate(mon.name, s_l, w) for mon, s_l, w in gates]
            # replan from where the plan says we should be (unless we deviated), so the
            # plan does not slide later with every bit of tracking lag
            t_ref = now - self.t_plan if self.plan is not None else 0.0
            s0, v0, _ = replan_origin(self.plan, t_ref, s, v, barriers=[s_l for _, s_l, _ in gates])
            try:
                self.future = self.pool.submit(worker.solve, s0, v0, g, self.s_goal, self.a_cmd)
                self._future_t = now
            except Exception as e:  # noqa: BLE001 - e.g. BrokenProcessPool: restart the worker
                self.get_logger().error(f"Planner process unavailable ({e}), restarting it")
                self._start_pool()
            self.t_last_submit = now

        # light nodes launched with wait_for_start:=true begin their cycle once we have a plan;
        # repeat (1 Hz) until every light runs, in case a light node starts after us
        if (self.start_lights and self.plan is not None and now - self.t_start_pub > 1.0
                and any(not m.running or m.state == "??" for m in self.monitors)):
            self.start_pub.publish(Empty())
            self.t_start_pub = now

        # 2) speed command
        if self.plan is None:
            v_cmd, s_p, v_p, a_p = 0.0, s, 0.0, 0.0
        else:
            v_cmd, s_p, v_p, a_p = self.tracker.command(self.plan, now - self.t_plan, s, v, self.DELTA_TIME)
        self.a_cmd = a_p

        # 3) red-light guard, independent of the optimizer
        v_cmd, who = self.guard.limit(s, v, v_cmd, [GuardLight(m.name, s_l, w) for m, s_l, w in gates])
        if who is not None:
            self.guard_ticks += 1
            self.get_logger().warn(f"Guard braking for {who}", throttle_duration_sec=1.0)

        # 4) goal
        if self.s_goal is not None and s >= self.s_goal - 0.1 and v < 0.05:
            if not self.finished:
                self.get_logger().info(f"Goal reached after {now - self.t0:.1f} s, "
                                       f"battery model {self.energy_J:.1f} J, ∫a²dt {self.acc_cost:.2f}")
            self.finished = True
        if self.finished:
            v_cmd = 0.0

        # 5) steering: pure pursuit on a look-ahead point of the route
        lookahead = min(max(0.8, 0.5 * v), 1.7) + 0.1 * v
        target = self.route.lookahead_point(s, lookahead)
        steering = self.controller.compute_steering((x, y, yaw, v), target)

        if self.safety_lock:
            steering, v_cmd = 0.0, 0.0
        self.actuation.send_control(steering, v_cmd if self.is_sim else -v_cmd)  # real car: inverted

        # 6) bookkeeping
        self.a_meas = 0.7 * self.a_meas + 0.3 * (v - self.v_prev) / self.DELTA_TIME
        self.v_prev = v
        dt = self.DELTA_TIME
        p = CarParams
        power = p.mass * self.a_meas * v + p.mass * p.g * p.c_rr * v + 0.5 * p.rho * p.cd_a * v ** 3
        self.energy_J += max(power, 0.0) * dt / p.eta
        self.acc_cost += self.a_meas ** 2 * dt
        t_rel = now - self.t0
        self.st_actual.append((t_rel, s))
        solve_ms = 1e3 * self.plan.solve_time if self.plan is not None else 0.0
        vals = dict(v_plan=v_p, v_cmd=v_cmd, v_meas=v, a_plan=a_p, s=s, s_plan=s_p, energy_J=self.energy_J,
                    acc_cost=self.acc_cost, solve_ms=solve_ms, guard=float(who is not None))
        for k, val in vals.items():
            self.telemetry[k].publish(Float32(data=float(val)))
        if self.log_writer is not None:
            row = dict(t=f"{t_rel:.3f}", x=f"{x:.3f}", y=f"{y:.3f}", s=f"{s:.3f}", s_plan=f"{s_p:.3f}",
                       v=f"{v:.3f}", v_plan=f"{v_p:.3f}", v_cmd=f"{v_cmd:.3f}", a_plan=f"{a_p:.3f}",
                       guard=int(who is not None), lidar_lock=int(self.safety_lock), solve_ms=f"{solve_ms:.1f}")
            for mon, s_l, _ in gates:
                row[f"{mon.name}_state"], row[f"{mon.name}_s"] = mon.state, f"{s_l:.3f}"
            for mon in self.monitors:  # lights already passed on an open route
                row.setdefault(f"{mon.name}_state", mon.state)
                row.setdefault(f"{mon.name}_s", f"{mon.s_local:.3f}")
            self.log_writer.writerow(row)
            if self.ticks % 20 == 0:
                self._log_file.flush()

        self._status(s, v, v_p, gates, now, who)

    def _collect_plan(self):
        if self.future is None or not self.future.done():
            return
        try:
            plan = self.future.result()
        except Exception as e:  # noqa: BLE001 - keep driving on the old plan, the guard keeps us safe
            self.get_logger().error(f"Planner failed: {e}")
            plan = None
            if "Broken" in type(e).__name__:
                self._start_pool()
        if plan is not None:
            old = self.plan
            keep_old = (not plan.ok and old is not None and old.ok
                        and time.monotonic() - self.t_plan < old.duration)
            if keep_old:
                # a failed replan does not throw away a plan we are still following
                # (the guard keeps us safe if that plan's crossing becomes unsafe)
                self.get_logger().warn("Replan infeasible, keeping the current plan", throttle_duration_sec=2.0)
            else:
                if plan.status == "failed":
                    self.get_logger().warn("No feasible plan, using comfortable stop", throttle_duration_sec=2.0)
                self.plan, self.t_plan = plan, self._future_t
        self.future = None

    def _status(self, s, v, v_p, gates, now, who):
        parts = [f"s={s:5.1f} m  v={v:4.2f} (plan {v_p:4.2f}) m/s"]
        if self.plan is not None:
            parts.append(f"{self.plan.status}, solve {1e3 * self.plan.solve_time:.0f} ms")
        ahead = sorted(gates, key=lambda g: g[1])
        if len(ahead) > 6:  # load tests: only the next few lights
            ahead = ahead[:4]
            parts.append(f"{len(gates)} lights ahead")
        for mon, s_l, w in ahead:
            tl = max(0.0, mon.time_left - (now - mon.t_left))
            eta = self.plan.time_at(s_l) - (now - self.t_plan) if self.plan is not None else math.inf
            win = self.plan.crossings.get(mon.name) if self.plan is not None else None
            if win is not None:
                win_txt = f"→ green #{win.index}"
            elif s_l > s + self.cfg.horizon:
                win_txt, eta = "beyond horizon", math.inf
            elif self.plan is not None and self.plan.status in ("hold", "failed"):
                win_txt = "stop before it"
            else:
                win_txt = ""
            if not w:
                win_txt += " (no guaranteed green known)"
            parts.append(f"{mon.name}: {mon.state} {tl:3.1f}s  ETA {eta:4.1f}s {win_txt}")
        if who:
            parts.append(f"GUARD({who})")
        if self.safety_lock:
            parts.append("LIDAR LOCK")
        self._status_text = " | ".join(parts)
        self.status_pub.publish(String(data=self._status_text))

    ## Visualisation

    def publish_viz(self):
        if self.route is None:
            return
        now = time.monotonic()
        stamp = self.get_clock().now().to_msg()
        arr = MarkerArray()

        def marker(mid, mtype, ns):
            m = Marker()
            m.header.frame_id, m.header.stamp = "map", stamp
            m.ns, m.id, m.type, m.action = ns, mid, mtype, Marker.ADD
            m.pose.orientation.w = 1.0
            return m

        if self.plan is not None:
            # plan coloured by speed
            t_rel = now - self.t_plan
            m = marker(0, Marker.LINE_STRIP, "plan")
            m.scale.x = 0.12
            tt = np.linspace(max(0.0, t_rel), self.plan.duration, 80)
            ss, vv, _ = self.plan.sample(tt)
            xs, ys = self.route.xy(ss)
            for xi, yi, vi in zip(xs, ys, vv):
                m.points.append(Point(x=float(xi), y=float(yi), z=0.05))
                r, g, b = speed_rgba(float(vi), self.v_max)
                m.colors.append(ColorRGBA(r=r, g=g, b=b, a=0.95))
            arr.markers.append(m)

            # ghost car where the plan wants us to be right now
            s_g, _, _ = self.plan.sample(t_rel)
            gx, gy = self.route.xy(s_g)
            yaw = float(self.route.yaw(s_g))
            m = marker(1, Marker.CUBE, "ghost")
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(gx), float(gy), 0.08
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.5, 0.26, 0.12
            m.color = ColorRGBA(r=0.2, g=0.5, b=1.0, a=0.35)
            arr.markers.append(m)

            # space-time diagram for Foxglove (pose.x = time [s], pose.y = s [m])
            self.st_plan_pub.publish(self._st_path(self.t_plan - self.t0 + tt, ss, stamp))

        for i, (mon, (lx, ly)) in enumerate(zip(self.monitors, self._light_xy)):
            s_l = mon.s_local
            px, py = self.route.xy(s_l)
            yaw = float(self.route.yaw(s_l))
            state = mon.state if mon.windows(now) or mon.state in LIGHT_RGBA else "??"
            r, g, b = LIGHT_RGBA.get(state, LIGHT_RGBA["??"])
            m = marker(10 + i, Marker.CUBE, "stop_line")  # line across the corridor at the light
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(px), float(py), 0.02
            m.pose.orientation.z, m.pose.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            m.scale.x, m.scale.y, m.scale.z = 0.08, 1.4, 0.04
            m.color = ColorRGBA(r=r, g=g, b=b, a=0.9)
            arr.markers.append(m)
            m = marker(20 + i, Marker.TEXT_VIEW_FACING, "light_label")
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(lx), float(ly), 1.1
            tl = max(0.0, mon.time_left - (now - mon.t_left)) if math.isfinite(mon.t_left) else 0.0
            dense = len(self.monitors) > 6  # load tests: short labels, or the map drowns in text
            m.text = f"{tl:.1f}" if dense else f"{mon.name}  {state}  {tl:.1f}s"
            m.scale.z = 0.18 if dense else 0.3
            m.color = ColorRGBA(r=r, g=g, b=b, a=1.0)
            arr.markers.append(m)

        if hasattr(self, "_status_text"):
            x, y, _, _ = self.localizer.get_state()
            m = marker(30, Marker.TEXT_VIEW_FACING, "status")
            m.pose.position.x, m.pose.position.y, m.pose.position.z = float(x), float(y), 0.9
            m.text = self._status_text.split(" | ")[0]
            m.scale.z = 0.22
            m.color = ColorRGBA(r=1.0, g=1.0, b=1.0, a=1.0)
            arr.markers.append(m)

        self.viz_pub.publish(arr)
        if self.st_actual:
            t, s = zip(*self.st_actual[-600:])
            self.st_actual_pub.publish(self._st_path(t, s, stamp))

    @staticmethod
    def _st_path(t, s, stamp):
        p = Path()
        p.header.frame_id, p.header.stamp = "st_diagram", stamp
        for ti, si in zip(t, s):
            ps = PoseStamped()
            ps.header = p.header
            ps.pose.position.x, ps.pose.position.y = float(ti), float(si)
            ps.pose.orientation.w = 1.0
            p.poses.append(ps)
        return p

    ## Tutorial 1 safety lock (unchanged logic)

    def lidar_callback(self, msg):
        vel = self.localizer.get_state()[3]
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + np.arange(len(ranges)) * msg.angle_increment
        thresholds = (1 - np.abs(angles) / msg.angle_max) * (self.k_th + self.k_v * vel) + self.c
        count = int(np.sum(ranges < thresholds))
        if self.safety_lock and count == 0:
            self.get_logger().info("Safety lock deactivated.")
            self.safety_lock = False
        elif not self.safety_lock and count > self.points_limit:
            self.get_logger().info("Safety lock activated.")
            self.safety_lock = True

    def on_shutdown(self):
        try:
            self.actuation.send_control(0.0, 0.0)
            if self.pool is not None:
                self.pool.shutdown(wait=False, cancel_futures=True)
            if self.log_writer is not None:
                self._log_file.close()
        except Exception:  # noqa: BLE001
            pass


if __name__ == "__main__":
    eco_drive.main()

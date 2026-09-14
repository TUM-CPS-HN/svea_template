#! /usr/bin/env python3

r"""ACC follower starter script for SVEA.

This is a skeleton, not a working example: copy it into your own package's
`scripts/` folder and fill in the TODOs. The path-following boilerplate
(steering via PurePursuitController, updating the goal/trajectory) is the
same as `pure_pursuit.py` from Tutorial 1 and is already filled in below —
what's missing is the Adaptive Cruise Control part: tracking the leader's
state and turning it into a target velocity.

See docs/tutorials/3_adaptive_cruise_control.md, Task 1, for the
requirements your control law needs to satisfy.
"""

import time

import numpy as np
from nav_msgs.msg import Odometry

from svea_core.interfaces import LocalizationInterface, ActuationInterface, ShowMarker, ShowPath
from svea_core.controllers.pure_pursuit import PurePursuitController
from svea_core import rosonic as rx


class acc_follower(rx.Node):

    DELTA_TIME = 0.1
    TRAJ_LEN = 20

    # Path (should match the leader's `points`, so both vehicles drive the same loop)
    points = rx.Parameter([-2.3, -7.1, 10.5, 11.7, 5.7, 15.0, -7.0, -4.0])
    is_sim = rx.Parameter(True)

    # --- ACC parameters shared by any control law; tune from the launch file ---
    leader_odom_topic = rx.Parameter('/svea_a/odometry/local')
    cruise_velocity = rx.Parameter(0.6)      # [m/s] speed to converge to when the gap is large
    max_velocity = rx.Parameter(1.0)         # [m/s]
    max_accel = rx.Parameter(0.4)            # [m/s^2]
    max_decel = rx.Parameter(0.8)            # [m/s^2]
    leader_timeout = rx.Parameter(1.0)       # [s] fall back if leader odom is older than this

    # TODO: add whatever parameters your chosen control law needs on top of
    # these (gains, distance/time thresholds, ...) — see Task 1.

    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()

    goal_marker = ShowMarker()
    path = ShowPath()

    # TODO: declare a subscriber for the leader's odometry.
    #
    # Hint: LocalizationInterface (above) always resolves to *your own*
    # vehicle's namespace (see the tutorial's namespacing discussion), so it
    # can't be reused to see the leader. You need a plain rx.Subscriber
    # pointed at `leader_odom_topic` instead:
    #
    # leader_sub = rx.Subscriber(Odometry, leader_odom_topic)
    #
    # @leader_sub
    # def _on_leader_odom(self, msg: Odometry) -> None:
    #     # store what you need from msg (position, velocity) plus
    #     # time.time(), so compute_acc_velocity() can check staleness
    #     ...

    def on_startup(self):
        """Same setup as pure_pursuit.py: controller, first goal, trajectory."""
        self._points = np.array(self.points).reshape(-1, 2).tolist()

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.cruise_velocity

        # TODO: initialize whatever state you need to track the leader, e.g.
        # self._leader_odom = None
        # self._leader_last_seen = None

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.curr = 0
        self.goal = self._points[self.curr]
        self.goal_marker.place([*self.goal, 0.5], color='green')
        self.update_traj(x, y)
        self.actuation.enable_difflock()

        self.create_timer(self.DELTA_TIME, self.loop)

    def compute_acc_velocity(self, state) -> float:
        """TODO: implement your ACC control law.

        See docs/tutorials/3_adaptive_cruise_control.md, Task 1, for the
        requirements it needs to satisfy. Use `state` for your own
        position/velocity, and whatever you stored in `_on_leader_odom` for
        the leader's. Don't forget to:
          - clip the result to [0, self.max_velocity]
          - rate-limit against the controller's current target_velocity
            using self.max_accel / self.max_decel, so commands don't jump
          - fall back to something safe if the leader hasn't been heard
            from in more than self.leader_timeout seconds
        """
        raise NotImplementedError("Task 1: implement the ACC control law")

    def loop(self):
        """Main control loop: same shape as pure_pursuit.py's loop()."""
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self.controller.is_finished:
            self.update_goal()
            self.update_traj(x, y)

        # TODO: this is the one line that makes it "adaptive" instead of a
        # constant-speed pure pursuit follower.
        self.controller.target_velocity = self.compute_acc_velocity(state)

        steering, velocity = self.controller.compute_control(state)
        self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        if self.is_sim:
            self.actuation.send_control(steering, velocity)
        else:
            self.actuation.send_control(steering, -1 * velocity)  # Invert velocity for real-world operation

    def update_goal(self):
        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        self.goal_marker.place([*self.goal, 0.5], color='green')

    def update_traj(self, x, y):
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys
        self.path.publish_path(xs, ys)


if __name__ == '__main__':
    acc_follower.main()

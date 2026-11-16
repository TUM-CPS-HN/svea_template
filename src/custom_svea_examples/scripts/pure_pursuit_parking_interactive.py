#!/usr/bin/env python3

import numpy as np

from custom_svea_core.interfaces import LocalizationInterface
from custom_svea_core.controllers.pure_pursuit import PurePursuitController
from custom_svea_core.controllers.mpc_park import MPCParkingController
from custom_svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
from custom_svea_core import rosonic as rx

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point, PointStamped


class pure_pursuit_parking_interactive(rx.Node):
    """
    Combined interactive node for testing both Pure Pursuit path tracking
    and MPC reverse parking using '/clicked_point' in Foxglove:
      - Clicking in FRONT of the car (|angle_car| <= 90 deg) -> Pure Pursuit tracking
      - Clicking BEHIND the car (|angle_car| > 90 deg)       -> MPC reverse parking
    """

    PP_DELTA_TIME = 0.1       # Pure Pursuit control period [s]
    MPC_DELTA_TIME = 0.4      # MPC Parking control period [s]
    TRAJ_LEN = 20

    LIDAR_OFFSET = 0.37       # Distance from car position to LiDAR mount [m]
    SAMPLE_SPREAD_DEG = 10.0  # Angular spread (each side) for sampling wall points [deg]

    target_velocity = rx.Parameter(0.5)
    is_sim = rx.Parameter(True)

    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()

    goal_marker = ShowMarker()
    path = ShowPath()

    def on_startup(self):
        # Initialize both controllers
        self.pp_controller = PurePursuitController()
        self.pp_controller.target_velocity = self.target_velocity

        self.mpc_controller = MPCParkingController(self)
        self.current_steering = 0.0

        # Mode can be "IDLE", "PURE_PURSUIT", or "MPC_PARKING"
        self.mode = "IDLE"

        # LiDAR state
        self.lidar_scan = []
        self.lidar_angle_min = None
        self.lidar_angle_inc = None
        self._logged_first_scan = False

        # Pure pursuit goal
        self.goal = [0.0, 0.0]

        # MPC parking state
        self.wall_line = None
        self.target_on_wall = None

        # Safety lock parameters (used during forward Pure Pursuit driving)
        self.safety_lock = False
        self.k_th = 0.15
        self.k_v = 0.35
        self.c = 0.2
        self.points_limit = 5

        self.actuation.enable_difflock()

        # Timers for each controller at its own sampling rate
        self.create_timer(self.PP_DELTA_TIME, self.pp_loop)
        self.create_timer(self.MPC_DELTA_TIME, self.mpc_loop)

        scan_topic = f'/{"svea_a" if self.is_sim else "self"}/scan'
        self.scan_subscriber = self.create_subscription(
            LaserScan,
            scan_topic,
            self.lidar_callback,
            1
        )

        self.clicked_point_subscriber = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            1
        )

        self.safe_zone_pub = self.create_publisher(Marker, '/safe_zone_visual', 10)

        self.get_logger().info(
            f"pure_pursuit_parking_interactive started (is_sim={self.is_sim}). "
            f"Subscribed to '{scan_topic}' and '/clicked_point'. "
            f"Click in front of the car for Pure Pursuit, or behind the car for MPC Parking."
        )

    def lidar_callback(self, msg):
        self.lidar_scan = msg.ranges
        self.lidar_angle_min = msg.angle_min
        self.lidar_angle_inc = msg.angle_increment

        if not self._logged_first_scan:
            self._logged_first_scan = True
            self.get_logger().info(
                f"First LiDAR scan received: {len(self.lidar_scan)} beams, "
                f"angle_min={np.degrees(self.lidar_angle_min):.1f} deg, "
                f"angle_inc={np.degrees(self.lidar_angle_inc):.2f} deg"
            )

        # Turn off the LiDAR safety lock mechanism and delete the visual marker while MPC parking
        if self.mode == "MPC_PARKING":
            self.safety_lock = False
            marker = Marker()
            marker.header.frame_id = msg.header.frame_id
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'safe_zone'
            marker.id = 0
            marker.action = Marker.DELETE
            self.safe_zone_pub.publish(marker)
            return

        state = self.localizer.get_state()
        x, y, yaw, vel = state
        ranges = np.array(msg.ranges)
        angles = msg.angle_min + (np.arange(len(ranges)) * msg.angle_increment)
        thresholds = (1 - (np.abs(angles) / msg.angle_max)) * (self.k_th + self.k_v * max(0.0, vel)) + self.c

        condition_met = ranges < thresholds
        count = np.sum(condition_met)
        if self.safety_lock and count == 0:
            self.get_logger().info("Safety lock deactivated.")
            self.safety_lock = False
        elif not self.safety_lock and count > self.points_limit:
            self.get_logger().info("Safety lock activated.")
            self.safety_lock = True

        # --- Draw the Safe Zone in Foxglove ---
        marker = Marker()
        marker.header.frame_id = msg.header.frame_id
        marker.header.stamp = self.get_clock().now().to_msg()
        marker.ns = 'safe_zone'
        marker.id = 0
        marker.type = Marker.LINE_STRIP
        marker.action = Marker.ADD
        marker.scale.x = 0.05
        marker.color.a = 1.0
        marker.color.r = 1.0
        marker.color.g = 1.0
        marker.color.b = 0.0

        x_coords = thresholds * np.cos(angles)
        y_coords = thresholds * np.sin(angles)

        for px, py in zip(x_coords, y_coords):
            pt = Point()
            pt.x = float(px)
            pt.y = float(py)
            pt.z = 0.0
            marker.points.append(pt)

        self.safe_zone_pub.publish(marker)

    def parking_setup(self, target_parking_point: list, point_on_wall: np.ndarray, wall_line: list):
        wall_to_target = np.array(target_parking_point) - np.array(point_on_wall)

        self.wall_line = wall_line
        wall_unit_vector = np.array([-self.wall_line[1], self.wall_line[0]])
        wall_unit_vector = wall_unit_vector / np.linalg.norm(wall_unit_vector)

        proj_len = wall_unit_vector @ wall_to_target
        self.target_on_wall = np.array(point_on_wall) + proj_len * wall_unit_vector

        # Visualize projected parking target on wall in red/orange to distinguish from Pure Pursuit
        self.goal_marker.place(
            [float(self.target_on_wall[0]), float(self.target_on_wall[1]), 0.2],
            color='red',
            scale=0.2
        )

        wall_angle_deg = np.degrees(np.arctan2(wall_unit_vector[1], wall_unit_vector[0]))
        self.get_logger().info(
            f"[Parking Setup] wall_line=[A={wall_line[0]:.3f}, B={wall_line[1]:.3f}, C={wall_line[2]:.3f}], "
            f"wall_unit_vec=[{wall_unit_vector[0]:.3f}, {wall_unit_vector[1]:.3f}] ({wall_angle_deg:.1f} deg), "
            f"proj_len={proj_len:.3f} m -> target_on_wall=[{self.target_on_wall[0]:.3f}, {self.target_on_wall[1]:.3f}]"
        )

    def two_points_to_line_eq(self, p1: np.ndarray, p2: np.ndarray):
        dx, dy = p2 - p1
        norm = np.hypot(dx, dy)

        if norm < 1e-6:
            raise ValueError(f"Sampled wall points are too close (norm={norm:.6f}): p1={p1}, p2={p2}")

        A = dy / norm
        B = -dx / norm
        C = (dx * p1[1] - dy * p1[0]) / norm

        return [A, B, C]

    def clicked_point_callback(self, msg):
        clicked_xy = [msg.point.x, msg.point.y]
        state = self.localizer.get_state()
        car_x, car_y, yaw, vel = state
        car_pos = np.array([car_x, car_y])

        # Determine whether the clicked point is in front of or behind the car
        car_to_click = np.array(clicked_xy) - car_pos
        click_angle_global = np.arctan2(car_to_click[1], car_to_click[0])
        click_angle_car = np.arctan2(
            np.sin(click_angle_global - yaw),
            np.cos(click_angle_global - yaw)
        )

        # If clicked in front of the car (|angle| <= 90 deg) -> Pure Pursuit
        if abs(click_angle_car) <= (np.pi / 2.0):
            self.goal = clicked_xy
            self.goal_marker.place([*self.goal, 0.5], color='blue', scale=0.3)
            self.update_traj(car_x, car_y)
            self.pp_controller.is_finished = False
            self.mode = "PURE_PURSUIT"
            self.get_logger().info(
                f"[Click -> Pure Pursuit] goal=({clicked_xy[0]:.2f}, {clicked_xy[1]:.2f}), "
                f"angle_car={np.degrees(click_angle_car):.1f} deg (in front of car)."
            )
            return

        # Otherwise (clicked behind the car, |angle| > 90 deg) -> MPC Parking
        if not self.lidar_scan or self.lidar_angle_min is None or self.lidar_angle_inc is None:
            self.get_logger().warn("Clicked behind car for MPC Parking, but no LiDAR scan available yet! Ignoring click.")
            return

        lidar_pos = car_pos + self.LIDAR_OFFSET * np.array([np.cos(yaw), np.sin(yaw)])
        lidar_to_target = np.array(clicked_xy) - lidar_pos
        target_vector_angle_global = np.arctan2(lidar_to_target[1], lidar_to_target[0])
        target_vector_angle_car = np.arctan2(
            np.sin(target_vector_angle_global - yaw),
            np.cos(target_vector_angle_global - yaw)
        )

        num_beams = len(self.lidar_scan)
        target_vector_index_car = round((target_vector_angle_car - self.lidar_angle_min) / self.lidar_angle_inc)

        self.get_logger().info(
            f"[Click -> MPC Parking] clicked=({clicked_xy[0]:.2f}, {clicked_xy[1]:.2f}), "
            f"car=({car_x:.2f}, {car_y:.2f}, yaw={np.degrees(yaw):.1f} deg), "
            f"lidar_pos=({lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}), "
            f"angle_global={np.degrees(target_vector_angle_global):.1f} deg, "
            f"angle_car={np.degrees(target_vector_angle_car):.1f} deg, "
            f"beam_idx={target_vector_index_car}/{num_beams - 1}"
        )

        index_offset = max(1, round(np.radians(self.SAMPLE_SPREAD_DEG) / self.lidar_angle_inc))

        if target_vector_index_car - index_offset < 0:
            wall_point1_index = 0
            wall_point2_index = min(num_beams - 1, index_offset)
            self.get_logger().info(
                f"Target is near/in right blind spot (beam_idx={target_vector_index_car}); "
                f"using extreme right beams [{wall_point1_index}, {wall_point2_index}]."
            )
        elif target_vector_index_car + index_offset >= num_beams:
            wall_point1_index = max(0, num_beams - 1 - index_offset)
            wall_point2_index = num_beams - 1
            self.get_logger().info(
                f"Target is near/in left blind spot (beam_idx={target_vector_index_car}); "
                f"using extreme left beams [{wall_point1_index}, {wall_point2_index}]."
            )
        else:
            wall_point1_index = target_vector_index_car - index_offset
            wall_point2_index = target_vector_index_car + index_offset

        wall_point1_dist = self.lidar_scan[wall_point1_index]
        wall_point2_dist = self.lidar_scan[wall_point2_index]

        if not (np.isfinite(wall_point1_dist) and np.isfinite(wall_point2_dist)):
            self.get_logger().error(
                f"Invalid LiDAR range(s) at sampled wall indices "
                f"[{wall_point1_index}]={wall_point1_dist}, [{wall_point2_index}]={wall_point2_dist} (nan/inf). "
                f"Try clicking closer to a visible wall."
            )
            return

        wall_point1_angle_global = yaw + self.lidar_angle_inc * wall_point1_index + self.lidar_angle_min
        wall_point2_angle_global = yaw + self.lidar_angle_inc * wall_point2_index + self.lidar_angle_min

        wall_point1_unit = np.array([np.cos(wall_point1_angle_global), np.sin(wall_point1_angle_global)])
        wall_point1_vec = wall_point1_unit * wall_point1_dist
        wall_point2_unit = np.array([np.cos(wall_point2_angle_global), np.sin(wall_point2_angle_global)])
        wall_point2_vec = wall_point2_unit * wall_point2_dist

        wall_point1 = lidar_pos + wall_point1_vec
        wall_point2 = lidar_pos + wall_point2_vec

        self.get_logger().info(
            f"[Wall Sample] idx1={wall_point1_index} (d={wall_point1_dist:.2f} m, pt=[{wall_point1[0]:.2f}, {wall_point1[1]:.2f}]), "
            f"idx2={wall_point2_index} (d={wall_point2_dist:.2f} m, pt=[{wall_point2[0]:.2f}, {wall_point2[1]:.2f}])"
        )

        try:
            wall_line = self.two_points_to_line_eq(wall_point1, wall_point2)
        except ValueError as e:
            self.get_logger().error(f"Failed to compute wall line: {e}")
            return

        self.parking_setup(clicked_xy, wall_point1, wall_line)
        self.mode = "MPC_PARKING"

    def pp_loop(self):
        """Pure Pursuit control loop (runs at PP_DELTA_TIME = 0.1 s when in PURE_PURSUIT mode)."""
        if self.mode != "PURE_PURSUIT":
            return

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self.pp_controller.is_finished:
            self.get_logger().info(f"[Pure Pursuit] Goal reached at x={x:.2f}, y={y:.2f}, yaw={np.degrees(yaw):.1f} deg.")
            self.actuation.send_control(0.0, 0.0)
            self.current_steering = 0.0
            self.mode = "IDLE"
            return

        steering, velocity = self.pp_controller.compute_control(state)
        if self.safety_lock:
            steering = 0.0
            velocity = 0.0

        self.current_steering = steering
        self.get_logger().info(
            f"[PP Loop] pos=({x:.2f}, {y:.2f}), yaw={np.degrees(yaw):.1f} deg, v={vel:.2f} | "
            f"cmd: steer={np.degrees(steering):.1f} deg ({steering:.2f} rad), vel={velocity:.2f} m/s"
        )

        if self.is_sim:
            self.actuation.send_control(steering, velocity)
        else:
            self.actuation.send_control(steering, -1 * velocity)

    def mpc_loop(self):
        """MPC Parking control loop (runs at MPC_DELTA_TIME = 0.4 s when in MPC_PARKING mode)."""
        if self.mode != "MPC_PARKING":
            return

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        full_state = [x, y, yaw, vel, self.current_steering]

        steering_cmd, velocity_cmd = self.mpc_controller.compute_control(
            full_state,
            self.target_on_wall,
            self.wall_line
        )

        optimal_states = self.mpc_controller.get_optimal_states()
        if optimal_states is not None:
            xs = optimal_states[0, :].tolist()
            ys = optimal_states[1, :].tolist()
            self.path.publish_path(xs, ys)

        self.current_steering = steering_cmd

        # Compute rear axle position and errors for debugging logs
        rear_x = x - 0.15 * np.cos(yaw)
        rear_y = y - 0.15 * np.sin(yaw)
        A, B, C = self.wall_line
        wall_norm = np.hypot(A, B) + 1e-6
        wall_unit = np.array([-B, A]) / wall_norm
        car_vec = np.array([np.cos(yaw), np.sin(yaw)])
        if np.dot(car_vec, wall_unit) < 0:
            wall_unit = -wall_unit
        wall_angle = np.arctan2(wall_unit[1], wall_unit[0])
        yaw_err = np.arctan2(np.sin(wall_angle - yaw), np.cos(wall_angle - yaw))
        pos_err = np.array([rear_x - self.target_on_wall[0], rear_y - self.target_on_wall[1]])
        long_err = abs(float(wall_unit @ pos_err))
        lat_dist = abs(A * rear_x + B * rear_y + C) / wall_norm

        self.get_logger().info(
            f"[MPC Loop] rear=({rear_x:.2f}, {rear_y:.2f}), yaw={np.degrees(yaw):.1f} deg, v={vel:.2f} | "
            f"err: long={long_err:.2f} m, lat={lat_dist:.2f} m, yaw={np.degrees(yaw_err):.1f} deg ({yaw_err:.2f} rad) | "
            f"cmd: steer={np.degrees(steering_cmd):.1f} deg ({steering_cmd:.2f} rad), vel={velocity_cmd:.2f} m/s"
        )

        if self.is_sim:
            self.actuation.send_control(steering_cmd, velocity_cmd)
        else:
            self.actuation.send_control(steering_cmd, -1 * velocity_cmd)

    def update_traj(self, x, y):
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.pp_controller.traj_x = xs
        self.pp_controller.traj_y = ys
        self.path.publish_path(xs, ys)


if __name__ == '__main__':
    pure_pursuit_parking_interactive.main()

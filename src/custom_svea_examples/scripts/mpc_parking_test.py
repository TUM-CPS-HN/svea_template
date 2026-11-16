#!/usr/bin/env python3

import math
import numpy as np

from custom_svea_core.interfaces import LocalizationInterface
from custom_svea_core.controllers.mpc_park import MPCParkingController
from custom_svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
from custom_svea_core import rosonic as rx

from geometry_msgs.msg import PointStamped
from sensor_msgs.msg import LaserScan

class mpc_parker(rx.Node):
    """
    Note: this assumes you click in foxglove in a position that is actually close to the wall.
    """

    # This the corresponding parameter in the MPC controller, make it so that you pass it.
    DELTA_TIME = 0.4  
    LIDAR_OFFSET = 0.37  # Distance from car position to LiDAR mount [m]
    SAMPLE_SPREAD_DEG = 10.0  # Angular spread (each side) for sampling wall points [deg]

    is_sim = rx.Parameter(True)
    
    # Interfaces
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    
    goal_marker = ShowMarker() 
    path = ShowPath() 

    def on_startup(self):
        self.controller = MPCParkingController(self)
        self.lidar_scan = [] # Starts from rear right to rear left.
        self.lidar_angle_min = None
        self.lidar_angle_inc = None
        self._logged_first_scan = False

        # Track the steering angle, I can now see the problem with Ali's work.
        self.current_steering = 0.0

        self.actuation.enable_difflock() 

        self.parking_mode = False
        self.loop_timer = None

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

        self.get_logger().info(
            f"mpc_parker started (is_sim={self.is_sim}). "
            f"Subscribed to '{scan_topic}' and '/clicked_point'. Waiting for click..."
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

    def parking_setup(self, target_parking_point: list, point_on_wall: np.ndarray, wall_line: list):
        wall_to_target = np.array(target_parking_point) - np.array(point_on_wall)

        self.wall_line = wall_line
        wall_unit_vector = np.array([-self.wall_line[1], self.wall_line[0]])
        wall_unit_vector = wall_unit_vector / np.linalg.norm(wall_unit_vector)

        proj_len = wall_unit_vector @ wall_to_target # can be negative
        self.target_on_wall = np.array(point_on_wall) + proj_len * wall_unit_vector

        # Visualize projected target on wall in Foxglove
        self.goal_marker.place([float(self.target_on_wall[0]), float(self.target_on_wall[1]), 0.2], color='blue', scale=0.2)

        wall_angle_deg = np.degrees(np.arctan2(wall_unit_vector[1], wall_unit_vector[0]))
        self.get_logger().info(
            f"[Parking Setup] wall_line=[A={wall_line[0]:.3f}, B={wall_line[1]:.3f}, C={wall_line[2]:.3f}], "
            f"wall_unit_vec=[{wall_unit_vector[0]:.3f}, {wall_unit_vector[1]:.3f}] ({wall_angle_deg:.1f} deg), "
            f"proj_len={proj_len:.3f} m -> target_on_wall=[{self.target_on_wall[0]:.3f}, {self.target_on_wall[1]:.3f}]"
        )

    def two_points_to_line_eq(self, p1: np.ndarray, p2: np.ndarray): # line equation: Ax + By + C = 0
        dx, dy = p2 - p1
        norm = np.hypot(dx, dy)
        
        if norm < 1e-6:
            raise ValueError(f"Sampled wall points are too close (norm={norm:.6f}): p1={p1}, p2={p2}")
            
        A = dy / norm
        B = -dx / norm
        C = (dx * p1[1] - dy * p1[0]) / norm
        
        return [A, B, C]

    def clicked_point_callback(self, msg):
        if not self.lidar_scan or self.lidar_angle_min is None or self.lidar_angle_inc is None:
            self.get_logger().warn("Clicked point received, but no LiDAR scan available yet! Ignoring click.")
            return

        # Deduce the data needed from lidar points here.
        target_parking_point = [msg.point.x, msg.point.y]
        state = self.localizer.get_state()
        car_x, car_y, yaw, vel = state
        car_pos = np.array([car_x, car_y])

        # Account for LiDAR mount offset forward along car heading
        lidar_pos = car_pos + self.LIDAR_OFFSET * np.array([np.cos(yaw), np.sin(yaw)])

        lidar_to_target = np.array(target_parking_point) - lidar_pos
        target_vector_angle_global = np.arctan2(lidar_to_target[1], lidar_to_target[0])
        # Wrap relative angle into [-pi, pi]
        target_vector_angle_car = np.arctan2(
            np.sin(target_vector_angle_global - yaw),
            np.cos(target_vector_angle_global - yaw)
        )

        num_beams = len(self.lidar_scan)
        target_vector_index_car = round((target_vector_angle_car - self.lidar_angle_min) / self.lidar_angle_inc)

        self.get_logger().info(
            f"[Click] clicked=({target_parking_point[0]:.2f}, {target_parking_point[1]:.2f}), "
            f"car=({car_x:.2f}, {car_y:.2f}, yaw={np.degrees(yaw):.1f} deg), "
            f"lidar_pos=({lidar_pos[0]:.2f}, {lidar_pos[1]:.2f}), "
            f"angle_global={np.degrees(target_vector_angle_global):.1f} deg, "
            f"angle_car={np.degrees(target_vector_angle_car):.1f} deg, "
            f"beam_idx={target_vector_index_car}/{num_beams - 1}"
        )

        # Compute index offset from angular spread (e.g. 10 deg -> 5 indices at 2 deg/beam, or 40 at 0.25 deg/beam)
        index_offset = max(1, round(np.radians(self.SAMPLE_SPREAD_DEG) / self.lidar_angle_inc))

        # If the target is in or near the rear blind spot, sample the extreme beam on that side
        # and a beam `index_offset` inside the FOV so the two points are always distinct and on the same side.
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

        # Reconstruct global wall points from the LiDAR origin
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

        self.parking_setup(target_parking_point, wall_point1, wall_line)
        self.parking_mode = True

        # Only create the control loop timer once (avoid duplicate timers on multiple clicks)
        if self.loop_timer is None:
            self.loop_timer = self.create_timer(self.DELTA_TIME, self.loop)

    def loop(self):
        if not self.parking_mode:
            return

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        # TODO: think about a stopping condition

        full_state = [x, y, yaw, vel, self.current_steering]

        steering_cmd, velocity_cmd = self.controller.compute_control(
            full_state, 
            self.target_on_wall, 
            self.wall_line
        )

        optimal_states = self.controller.get_optimal_states()
        if optimal_states is not None:
            # optimal_states is a numpy array of shape (5, N+1)
            # Row 0 is X, Row 1 is Y. Convert them to standard Python lists for the interface.
            xs = optimal_states[0, :].tolist()
            ys = optimal_states[1, :].tolist()

            self.path.publish_path(xs, ys)

        # Update steering tracker for the next loop's state initialization
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

if __name__ == '__main__':
    mpc_parker.main()

# TODO: integrate this into the pure_pursuit labs script so that it parks at some point of it route.
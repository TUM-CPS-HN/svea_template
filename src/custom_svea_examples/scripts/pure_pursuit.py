#! /usr/bin/env python3

import numpy as np

from custom_svea_core.interfaces import LocalizationInterface
from custom_svea_core.controllers.pure_pursuit import PurePursuitController
from custom_svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
from custom_svea_core import rosonic as rx

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


class pure_pursuit(rx.Node):

    r"""Pure Pursuit example script for SVEA.

    #**Background**

    This script implements a simple Pure Pursuit controller that follows a
    predefined path. The path is defined by a set of points, and the controller
    computes the steering angle and velocity to follow the path.

    The script also includes visualization of the goal and the path being
    followed.

    #**Preparation**

    TODO: Add instructions for setting up the teleoperation environment.

    #**Simulation**

    To run the Pure Pursuit example in simulation, you can use the following command:
    ```bash
    ros2 launch custom_svea_examples floor2.xml is_sim:=true
    ```
    This launch file includes the following components, with example parameters:

        # Initial state of the robot (x, y, yaw, velocity)
        state:=[-7.4, -15.3, 0.9, 0.0] 
        # Points defining the path to follow. Each point is a string representation of a list.
        points:=['[-2.3,-7.1]','[10.5,11.7]','[5.7,15.0]','[-7.0,-4.0]'] 

    Attributes:
        points: List of points defining the path to follow.
        actuation: Actuation interface for sending control commands.
        localizer: Localization interface for receiving state information.
        goal_mark: ShowMarker for visualizing the goal.
        path: ShowPath for visualizing the path.
    """

    DELTA_TIME = 0.1  # units?? I guess seconds.
    TRAJ_LEN = 20
    WAYPOINT_SWITCH_DISTANCE = 0.55  # [m] remaining along the active leg

    points = rx.Parameter([-2.3, -7.1, 10.5, 11.7, 5.7, 15.0, -7.0, -4.0])
    target_velocity = rx.Parameter(0.5)
    is_sim = rx.Parameter(True)
    
    # Interfaces
    
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    
    goal_marker = ShowMarker() # for goal visualization
    path = ShowPath() # for path visualization

    def on_startup(self):
        """Set up the controller and wait for the first localized position."""
        # Convert parameter to numerical list
        self._points = np.array(self.points).reshape(-1, 2).tolist()

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        self.curr = 0
        self.goal = self._points[self.curr]
        self.segment_start = None
        self.localizer.add_callback(self.initialize_route)
        self.actuation.enable_difflock() 

        self.create_timer(self.DELTA_TIME, self.loop)

        # safety lock parameters
        self.safety_lock = False
        self.k_th = 0.15  
        self.k_v = 0.35   
        self.c = 0.2
        self.points_limit = 5

        self.scan_subscriber = self.create_subscription(
            LaserScan,
            f'/{"svea_a" if self.is_sim else "self"}/scan',
            self.lidar_callback,
            1
        )

        self.safe_zone_pub = self.create_publisher(Marker, '/safe_zone_visual', 10)

    def initialize_route(self, odometry):
        if self.segment_start is None:
            x, y, _, _ = self.localizer.get_state(odometry)
            self.goal_marker.place([*self.goal, 0.5], color='blue')
            self.update_traj(x, y)

    def lidar_callback(self, msg):
            # self.get_logger().info(f"Total points received: {len(msg.ranges)}")
            state = self.localizer.get_state()
            x, y, yaw, vel = state
            ranges = np.array(msg.ranges)
            angles = msg.angle_min + (np.arange(len(ranges)) * msg.angle_increment)
            thresholds = (1 - (np.abs(angles) / msg.angle_max)) * (self.k_th + self.k_v * vel) + self.c

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
            marker.header.frame_id = msg.header.frame_id # Ties the drawing to the LiDAR's position
            marker.header.stamp = self.get_clock().now().to_msg()
            marker.ns = 'safe_zone'
            marker.id = 0
            marker.type = Marker.LINE_STRIP
            marker.action = Marker.ADD
            
            # Set line thickness
            marker.scale.x = 0.05 
            
            # Set color to Yellow (RGBA: 1.0, 1.0, 0.0, 1.0)
            marker.color.a = 1.0
            marker.color.r = 1.0
            marker.color.g = 1.0
            marker.color.b = 0.0
            
            # Convert polar arrays (angle, threshold radius) to Cartesian arrays (x, y)
            x_coords = thresholds * np.cos(angles)
            y_coords = thresholds * np.sin(angles)
            
            # Populate the marker points
            for x, y in zip(x_coords, y_coords):
                pt = Point()
                pt.x = float(x)
                pt.y = float(y)
                pt.z = 0.0
                marker.points.append(pt)
                
            self.safe_zone_pub.publish(marker)

    def loop(self):
        """Track a continuous target on the active waypoint leg."""
        if self.segment_start is None:
            return

        state = self.localizer.get_state()
        x, y, yaw, vel = state
        position = np.array([x, y])
        progress = self.progress_along_segment(position)

        # Use distance along the route so the spacing of the visualization
        # points cannot affect where the vehicle starts turning.
        switch_distance = min(self.WAYPOINT_SWITCH_DISTANCE, self.segment_length / 2)
        if self.segment_length - progress <= switch_distance:
            prev_goal = self.goal
            self.update_goal()
            self.update_traj(*prev_goal)
            progress = self.progress_along_segment(position)

        if self.segment_length == 0.0:
            steering, velocity = 0.0, 0.0
        else:
            lookahead = self.controller.Lfc + self.controller.k * vel
            target_distance = min(progress + lookahead, self.segment_length)
            target = self.segment_start + target_distance * self.segment_direction
            steering, velocity = self.controller.compute_control(state, target=target)
        self.get_logger().info(f"x: {x}, y: {y}, yaw: {yaw}, vel: {vel}, str_cmd: {steering}, vel_cmd: {velocity}")
        if self.safety_lock:
            steering = 0.0
            velocity = 0.0
        if self.is_sim:
            self.actuation.send_control(steering, velocity)
        else:
            self.actuation.send_control(steering, -1 * velocity)  # Invert velocity for real-world operation

    def update_goal(self):
        """
        Update the goal to the next point in the path. If the end of the path
        is reached, it wraps around to the beginning. The current index is
        incremented, and the goal marker is updated.
        """
        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.goal_marker.place([*self.goal, 0.5], color='blue')

    def progress_along_segment(self, position):
        return max(0.0, float((position - self.segment_start) @ self.segment_direction))

    def update_traj(self, x, y):
        """Set the active leg and publish samples of it for visualization."""
        start = np.array([x, y])
        displacement = np.asarray(self.goal) - start
        self.segment_length = float(np.linalg.norm(displacement))
        self.segment_direction = displacement
        if self.segment_length:
            self.segment_direction = displacement / self.segment_length
        self.segment_start = start

        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.path.publish_path(xs, ys)


if __name__ == '__main__':
    pure_pursuit.main()

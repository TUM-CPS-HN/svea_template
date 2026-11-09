#! /usr/bin/env python3

import numpy as np

from custom_svea_core.interfaces import LocalizationInterface
from custom_svea_core.controllers.pure_pursuit import PurePursuitController
from custom_svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
from custom_svea_core import rosonic as rx

from sensor_msgs.msg import LaserScan
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from geometry_msgs.msg import PointStamped


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

    target_velocity = rx.Parameter(0.5)
    is_sim = rx.Parameter(True)
    
    # Interfaces
    
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    
    goal_marker = ShowMarker() # for goal visualization
    path = ShowPath() # for path visualization

    def on_startup(self):
        """
        Initialize the Pure Pursuit controller and set up the path and goal.
        Controller is initialized with the target velocity and the points
        provided in the parameters. The current state is obtained from the
        localization interface, and the goal is set to the first point in the
        path.
        The trajectory is updated based on the current state and the goal.
        The controller is set to not finished initially, and a timer is created
        to call the loop method at regular intervals.
        """

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.goal = [x, y]
        self.goal_marker.place([*self.goal, 0.5], color='blue')
        self.update_traj(x, y)
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

        self.clicked_point_subscriber = self.create_subscription(
            PointStamped,
            '/clicked_point',
            self.clicked_point_callback,
            1
        )

        self.safe_zone_pub = self.create_publisher(Marker, '/safe_zone_visual', 10)

    def lidar_callback(self, msg):
            """
            Callback function for processing LiDAR scan data. It checks if the number
            of points within a certain threshold is above a limit, and activates or
            deactivates a safety lock accordingly.
            """
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

    def clicked_point_callback(self, msg):
        """
        Callback function for handling clicked points in the visualization.
        When a point is clicked, it updates the goal to the clicked point and
        updates the trajectory accordingly.
        """
        self.goal = [msg.point.x, msg.point.y]
        self.goal_marker.place([*self.goal, 0.5], color='blue')
        state = self.localizer.get_state()
        x, y, yaw, vel = state
        self.update_traj(x, y)
        self.controller.is_finished = False

    def loop(self):
        """
        Main loop of the Pure Pursuit controller. It retrieves the current state
        from the localization interface, computes the steering and velocity
        commands using the controller, and sends these commands to the actuation
        interface.
        If the controller has finished following the path, it updates the goal
        and trajectory based on the next point in the path.
        """
        state = self.localizer.get_state()
        x, y, yaw, vel = state

        if self.controller.is_finished:
            self.get_logger().info(f"Goal reached. x: {x}, y: {y}, yaw: {yaw}")
            return

        steering, velocity = self.controller.compute_control(state)
        self.get_logger().info(f"x: {x}, y: {y}, yaw: {yaw}, vel: {vel}, str_cmd: {steering}, vel_cmd: {velocity}")
        if self.safety_lock:
            steering = 0.0
            velocity = 0.0
        if self.is_sim:
            self.actuation.send_control(steering, velocity)
        else:
            self.actuation.send_control(steering, -1 * velocity)  # Invert velocity for real-world operation

    def update_traj(self, x, y):
        """
        Update the trajectory based on the current state and the goal. It
        generates a linear trajectory from the current position to the goal
        position, and updates the controller's trajectory points.
        The trajectory is visualized using the ShowPath interface.
        """
        xs = np.linspace(x, self.goal[0], self.TRAJ_LEN)
        ys = np.linspace(y, self.goal[1], self.TRAJ_LEN)
        self.controller.traj_x = xs
        self.controller.traj_y = ys
        self.path.publish_path(xs,ys)

if __name__ == '__main__':
    pure_pursuit.main()

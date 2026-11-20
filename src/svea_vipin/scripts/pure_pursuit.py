#! /usr/bin/env python3

import numpy as np

from svea_core.interfaces import LocalizationInterface
from svea_vipin.pure_pursuit_controller import PurePursuitController
from svea_core.interfaces import ActuationInterface, ShowMarker, ShowPath
from svea_core import rosonic as rx

from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point
from std_msgs.msg import ColorRGBA

#from sensor_msgs.msg import PointCloud
#from svea_core.utils.viz_util import publish_lidar_points



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
    ros2 launch svea_examples floor2.xml is_sim:=true
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

    DELTA_TIME = 0.1
    TRAJ_LEN = 20
    SWITCH_DIST = 1.5
    STOP_DIST = 0.8


    # x1, y1, x2, y2, x3, y3, x4, y4 is the format of the points parameter
    points = rx.Parameter([-2.3, -7.1, 10.5, 11.7, 5.7, 15.0, -7.0, -4.0])

    obstacles = rx.Parameter([4.1, 2.3])  # x, y coordinates of obstacles

    target_velocity = rx.Parameter(0.6)
    is_sim = rx.Parameter(True)
    
    # Interfaces
    
    actuation = ActuationInterface()
    localizer = LocalizationInterface()
    
    goal_marker = ShowMarker() # for goal visualization
    path = ShowPath() # for path visualization

    obstacle_pub = rx.Publisher(Marker, 'obstacles')


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
        # Convert parameter to numerical list
        self._points = np.array(self.points).reshape(-1, 2).tolist()
        self._obstacles = np.array(self.obstacles).reshape(-1, 2).tolist()

        self.controller = PurePursuitController()
        self.controller.target_velocity = self.target_velocity

        state = self.localizer.get_state()
        x, y, yaw, vel = state

        self.curr = 0
        self.goal = self._points[self.curr]
        self.goal_marker.place([*self.goal, 0.5], color='blue')
        self.update_traj(x, y)
        self.actuation.enable_difflock() 

        self.create_timer(self.DELTA_TIME, self.loop)

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
        #publish_lidar_points(self.obstacle_pub, self._obstacles)
        self._publish_obstacles()  # Publish obstacles for visualization

        if (self.controller.is_finished) or (self._distance_to_goal(x, y) < self.SWITCH_DIST):
            # here we need to remember the old goal before update_goal() runs, 
            # and pass its coordinates to update_traj instead of x, y.
            # because update goal overrites goal, and we want to update the trajectory from 
            # the old goal to the new goal.
            # start the new segment at the old corner, not at the car, 
            # so the path stays on the rectangle and doesn’t cut through the inner wall.
            
            prev_goal = self.goal
            self.update_goal()
            self.update_traj(prev_goal[0], prev_goal[1])

        steering, velocity = self.controller.compute_control(state)

        # now if the robot is within STOP_DIST of an obstacle, we stop the robot and put velocity = 0.0, and do not update the goal or trajectory until the robot is clear of the obstacle.
        if self._distance_to_obstacle(x, y) < self.STOP_DIST:
            self.get_logger().info("Obstacle detected! Stopping the robot.")
            self.actuation.send_control(0.0, 0.0)  # Stop the robot
            return  # Skip the rest of the loop until the robot is clear of the obstacle

        self.get_logger().info(f"Steering: {steering}, Velocity: {velocity}")
        if self.is_sim:
            self.actuation.send_control(steering, velocity)
        else:
            self.actuation.send_control(steering, -1 * velocity)  # Invert velocity for real-world operation

    def _publish_obstacles(self):
        """
        Publish the list of obstacles as Markers. 
        Each obstacle is represented as a point in the Marker message.
        """
        obstacle_marker_msg = Marker()
        obstacle_marker_msg.header.stamp = self.get_clock().now().to_msg()
        obstacle_marker_msg.header.frame_id = 'map'


        obstacle_marker_msg.type = Marker.SPHERE_LIST
        obstacle_marker_msg.action = Marker.ADD

        for obs in self._obstacles:
            obstacle_marker_msg.points.append(Point(x=obs[0], y=obs[1], z=0.0))

        obstacle_marker_msg.id = 0
        obstacle_marker_msg.color = ColorRGBA(r=1.0, g=0.0, b=0.0, a=1.0) # a = opacity, so make it fully visible
        obstacle_marker_msg.scale.x = 0.3
        obstacle_marker_msg.scale.y = 0.3
        obstacle_marker_msg.scale.z = 0.3
        obstacle_marker_msg.pose.orientation.w = 1.0

        self.obstacle_pub.publish(obstacle_marker_msg)

    def _distance_to_obstacle(self, x, y):
        """
        Calculate the Euclidean distance from the current position to the nearest obstacle.
        This is used to determine if the robot is close enough to an obstacle to take action.
        """
        if not self._obstacles:
            return float('inf')  # No obstacles, return infinite distance

        distances = [np.sqrt((obs[0] - x) ** 2 + (obs[1] - y) ** 2) for obs in self._obstacles]
        return min(distances)

    def _distance_to_goal(self, x, y):
        """
        Calculate the Euclidean distance from the current position to the goal.
        This is used to determine if the robot is close enough to the goal to
        switch to the next point in the path.
        """
        return np.sqrt((self.goal[0] - x) ** 2 + (self.goal[1] - y) ** 2)

    
    def update_goal(self):
        """
        Update the goal to the next point in the path. If the end of the path
        is reached, it wraps around to the beginning. The current index is
        incremented, and the goal marker is updated.
        """
        self.curr += 1
        self.curr %= len(self._points)
        self.goal = self._points[self.curr]
        self.controller.is_finished = False
        # Mark the goal
        self.goal_marker.place([*self.goal, 0.5], color='blue')

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

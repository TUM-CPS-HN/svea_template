#! /usr/bin/env python3

# imports
import random

from rclpy.clock import Clock, Duration
from std_msgs.msg import String, Float32, Empty

from svea_core import rosonic as rx
from svea_core.interfaces import ShowMarker


class traffic_light(rx.Node):

    ## Constants

    GREEN_TIME  = 5.                # [s]
    YELLOW_TIME = 2.                # [s]
    RED_TIME    = 5.                # [s]

    # dictionary for states
    TRANSITIONS = {
        'Rd': 'YG', # Red           -> Yellow-Green
        'YG': 'Gr', # Yellow-Green  -> Green
        'Gr': 'YR', # Green         -> Yellow-Red
        'YR': 'Rd', # Yellow-Red    -> Red
    }

    ## Parameters

    name = rx.Parameter('light_a')
    rate = rx.Parameter(10)
    alpha = rx.Parameter(0.0)

    pos_x = rx.Parameter(0.0)
    pos_y = rx.Parameter(0.0)
    wait_for_start = rx.Parameter(False)
    start_topic = rx.Parameter('/planned_trajectory/start')

    ## Publishers

    state_pub = rx.Publisher(String, '~/state')
    time_pub = rx.Publisher(Float32, '~/time_left')

    ## Interfaces

    light_marker = ShowMarker()

    ## Methods

    def on_startup(self):

        assert 0 <= self.alpha <= min(self.RED_TIME, self.GREEN_TIME)

        self._running = not self.wait_for_start
        self.switch('Rd')  # initial state is red
        if self.wait_for_start:
            self._start_sub = self.create_subscription(
                Empty,
                self.start_topic,
                self.start_callback,
                1,
            )
        self._loop_tmr = self.create_timer(1.0 / self.rate, self.loop)
        self.get_logger().info(f"Traffic light '{self.name}' fixed at ({self.pos_x}, {self.pos_y})")

    def start_callback(self, _msg):
        if self._running:
            return
        self._running = True
        self.switch('Rd')
        self.get_logger().info('Traffic light cycle started')

    def loop(self):

        now = Clock().now()

        # calculate time remaining for current state
        time_left = (self._time0 + self._delta - now).nanoseconds / 1e9

        # if time has elapsed, then switch to next state
        if self._running and time_left <= 0:
            self.switch()
            now = Clock().now()
            time_left = (self._time0 + self._delta - now).nanoseconds / 1e9
        elif not self._running:
            time_left = self.RED_TIME

        # publish current state and time remaining
        msg = String()
        msg.data = self._state
        self.state_pub.publish(msg)

        msg = Float32()
        msg.data = time_left
        self.time_pub.publish(msg)

        # visualization for the light
        self.visualize_traffic_light()

    def switch(self, to=None):
        """
        Change the traffic light to the next state
        """
        self._state = (to if to is not None else
                       self.TRANSITIONS[self._state])

        delta = (self.RED_TIME      if self._state == 'Rd' else
                 self.GREEN_TIME    if self._state  == 'Gr' else
                 self.YELLOW_TIME)
        if self._state in ('Rd', 'Gr'):
            delta += random.uniform(-self.alpha, +self.alpha)

        # delta determines how long each state lasts
        self._delta = Duration(seconds=delta)

        self._time0 = Clock().now()

    def visualize_traffic_light(self):
        """Publish the position of the traffic light for visualization in RViz/Foxglove."""

        color = {'Rd': 'red', 'Gr': 'green'}.get(self._state, 'yellow')
        self.light_marker.place([self.pos_x, self.pos_y, 0.5], color=color)


if __name__ == '__main__':

    traffic_light.main()

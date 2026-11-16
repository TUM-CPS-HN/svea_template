#! /usr/bin/env python3
"""Many traffic lights in one node (for load tests with dozens of lights).

Every light behaves exactly like ``traffic_light.py``: the same Rd -> YG -> Gr -> YR
cycle, the same ±alpha jitter on red and green, and the same topics
``/<name>/state`` (String) and ``/<name>/time_left`` (Float32), so the eco-drive
node cannot tell the difference. One process instead of one per light.

``offsets`` gives each light a phase offset: the time into its cycle, counted
from the start of red, at which it starts. With ``wait_for_start`` all lights
start on the first message on ``/planned_trajectory/start``.
"""

import random

from rclpy.clock import Clock
from std_msgs.msg import Empty, Float32, String
from visualization_msgs.msg import Marker, MarkerArray

from custom_svea_core import rosonic as rx


class traffic_light_group(rx.Node):

    GREEN_TIME = 5.0  # [s] same as traffic_light.py
    YELLOW_TIME = 2.0
    RED_TIME = 5.0
    TRANSITIONS = {'Rd': 'YG', 'YG': 'Gr', 'Gr': 'YR', 'YR': 'Rd'}
    COLORS = {'Rd': (1.0, 0.1, 0.1), 'Gr': (0.1, 0.9, 0.2)}  # yellow otherwise

    names = rx.Parameter(['light_00'])
    positions = rx.Parameter([0.0, 0.0])  # [x, y] per light (markers only)
    offsets = rx.Parameter([0.0])  # [s] time into the cycle at start, per light
    alpha = rx.Parameter(0.0)
    rate = rx.Parameter(10)
    seed = rx.Parameter(0)
    wait_for_start = rx.Parameter(False)
    start_topic = rx.Parameter('/planned_trajectory/start')

    def on_startup(self):
        n = len(self.names)
        assert len(self.offsets) == n and len(self.positions) == 2 * n
        assert 0 <= self.alpha <= min(self.RED_TIME, self.GREEN_TIME)
        self.rng = random.Random(self.seed)
        self.state = ['Rd'] * n
        self.t_end = [0.0] * n  # absolute end time of the current phase [s]
        self.running = False
        self.pubs = [(self.create_publisher(String, f'/{name}/state', 10),
                      self.create_publisher(Float32, f'/{name}/time_left', 10)) for name in self.names]
        self.marker_pub = self.create_publisher(MarkerArray, '/traffic_lights/markers', 1)
        if self.wait_for_start:
            self.create_subscription(Empty, self.start_topic, self.start_callback, 1)
        else:
            self.start()
        self.create_timer(1.0 / self.rate, self.loop)
        self.create_timer(0.2, self.publish_markers)
        self.get_logger().info(f'{n} traffic lights ready')

    def now(self):
        return Clock().now().nanoseconds / 1e9

    def nominal(self, state):
        return {'Rd': self.RED_TIME, 'Gr': self.GREEN_TIME}.get(state, self.YELLOW_TIME)

    def duration(self, state):
        d = self.nominal(state)
        if state in ('Rd', 'Gr'):
            d += self.rng.uniform(-self.alpha, self.alpha)
        return d

    def start(self):
        """Start every light ``offset`` seconds into its (nominal) cycle."""
        t = self.now()
        period = self.RED_TIME + self.GREEN_TIME + 2 * self.YELLOW_TIME
        for i, off in enumerate(self.offsets):
            state, remaining, off = 'Rd', self.RED_TIME, float(off) % period
            while off >= remaining:
                off -= remaining
                state = self.TRANSITIONS[state]
                remaining = self.nominal(state)
            self.state[i], self.t_end[i] = state, t + remaining - off
        self.running = True

    def start_callback(self, _msg):
        if not self.running:
            self.start()
            self.get_logger().info('Traffic light cycles started')

    def loop(self):
        t = self.now()
        for i, (state_pub, time_pub) in enumerate(self.pubs):
            if self.running:
                while t >= self.t_end[i]:
                    self.state[i] = self.TRANSITIONS[self.state[i]]
                    self.t_end[i] += self.duration(self.state[i])
                left = self.t_end[i] - t
            else:
                left = self.RED_TIME  # constant while waiting: the planner sees "not running"
            state_pub.publish(String(data=self.state[i]))
            time_pub.publish(Float32(data=float(left)))

    def publish_markers(self):
        arr = MarkerArray()
        stamp = self.get_clock().now().to_msg()
        for i, name in enumerate(self.names):
            m = Marker()
            m.header.frame_id, m.header.stamp = 'map', stamp
            m.ns, m.id, m.type, m.action = 'traffic_lights', i, Marker.SPHERE, Marker.ADD
            m.pose.position.x = float(self.positions[2 * i])
            m.pose.position.y = float(self.positions[2 * i + 1])
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = m.scale.y = m.scale.z = 0.3
            m.color.r, m.color.g, m.color.b = self.COLORS.get(self.state[i], (1.0, 0.8, 0.0))
            m.color.a = 1.0
            arr.markers.append(m)
        self.marker_pub.publish(arr)


if __name__ == '__main__':

    traffic_light_group.main()

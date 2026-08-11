#! /usr/bin/env python3

"""
single_encoder_twist_filter

Turns the unsigned cumulative distance published by the mavros wheel_odometry
plugin (one encoder per wheel) into a directional
geometry_msgs/TwistWithCovarianceStamped for robot_localization.

This is a drop-in replacement for mavros/wheel_odometry/velocity: same message
type, same semantics, but the sign of the motion is restored. Nothing is
integrated, so a momentary direction error only affects that one sample instead
of accumulating into a pose.

The IMU is used for exactly one thing here: deciding which way the car is
moving. It does not contribute to the published velocities -- linear velocity
comes from the wheels and yaw rate comes from the wheel difference. There is
therefore no double-counting against the EKF's existing imu0 input, and the
EKF's IMU configuration does not need to change.

------------------------------------------------------------------------------
Direction sources (direction_source)
------------------------------------------------------------------------------
    'manual_control'  mavros/manual_control/send (ManualControl)
                      z = 500 - velocity_percent*5, so z<500 is forward and
                      z>500 is reverse.
    'rc'              mavros/rc/in (RCIn), raw receiver channels.
    'auto'            RC takes priority, falls back to manual_control when RC
                      is stale or centred.
    'imu'             Trust only the sign of the IMU-integrated speed; hold the
                      previous direction when it is not trustworthy.
    'fusion'          Command-driven, but override with the IMU when the
                      integrated speed is large enough and disagrees.  <-- default

    Command-based direction is wrong during coast-down: the throttle has
    already reversed but the car is still rolling forward, so that motion would
    be reported as reverse travel. The sign of the IMU-integrated longitudinal
    speed reflects actual motion and closes that gap.

    Drift of the integrator is bounded by ZUPT (zero-velocity update): whenever
    the wheels report standstill the integrated speed is reset to zero and the
    accelerometer bias is re-estimated. The integrator only has to stay correct
    between two stops, so it never diverges without bound.

    Because of that online bias estimate, the raw IMU stream is good enough --
    no bias-removed topic is required. Note that the estimated bias also
    absorbs whatever gravity leaks into the x axis from a non-level mounting,
    which means the estimate is only valid for the pitch the vehicle had at the
    last standstill. On a slope the leaked gravity component appears as a
    constant acceleration and the integrated speed will drift until the next
    stop. Flat ground is assumed.

------------------------------------------------------------------------------
Model
------------------------------------------------------------------------------
    v = (ds_R + ds_L) / (2 * dt)
    w = (ds_R - ds_L) / (axle_track * dt)
"""

import math

from rclpy.qos import (
    QoSProfile,
    QoSDurabilityPolicy,
    QoSReliabilityPolicy,
    QoSHistoryPolicy,
)
from rclpy.time import Time

from sensor_msgs.msg import Imu
from geometry_msgs.msg import TwistWithCovarianceStamped
from mavros_msgs.msg import WheelOdomStamped, ManualControl, RCIn

from svea_core import rosonic as rx


# mavros publishes with sensor-data QoS, so subscribers must be BEST_EFFORT.
qos_subber = QoSProfile(
    reliability=QoSReliabilityPolicy.BEST_EFFORT,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=20,
)

# robot_localization subscribes with RELIABLE, so this publisher must match.
qos_pubber = QoSProfile(
    reliability=QoSReliabilityPolicy.RELIABLE,
    durability=QoSDurabilityPolicy.VOLATILE,
    history=QoSHistoryPolicy.KEEP_LAST,
    depth=10,
)


class single_encoder_twist_filter(rx.Node):
    """Unsigned cumulative distance + direction estimate -> body-frame twist."""

    ## ---------------- Parameters ---------------- ##

    # -- Topics --
    distance_topic = rx.Parameter('mavros/wheel_odometry/distance')
    twist_topic = rx.Parameter('wheel_odometry/twist/filtered')
    control_topic = rx.Parameter('mavros/manual_control/send')
    rc_topic = rx.Parameter('mavros/rc/in')
    # Relative name, so it resolves inside the vehicle namespace.
    # data_raw is fine here: only linear_acceleration.x is used, and its bias is
    # estimated online by the ZUPT below. The gyro is not touched at all.
    imu_topic = rx.Parameter('mavros/imu/data_raw')

    # Twist is expressed in the body frame, so this is the message frame_id.
    base_frame = rx.Parameter('base_link')

    # -- Vehicle geometry (see svea_core/params/mavros.yaml) --
    axle_track = rx.Parameter(0.32)
    left_index = rx.Parameter(0)
    right_index = rx.Parameter(1)

    # -- Direction estimation --
    direction_source = rx.Parameter('fusion')   # manual_control|rc|auto|imu|fusion
    default_direction = rx.Parameter(1)

    control_neutral = rx.Parameter(500.0)
    control_deadband = rx.Parameter(15.0)

    rc_channel = rx.Parameter(2)                # index into channels[], verify by echo
    rc_neutral = rx.Parameter(1500)
    rc_deadband = rx.Parameter(80)
    rc_reversed = rx.Parameter(False)
    rc_timeout = rx.Parameter(0.5)

    # Hysteresis for command-based direction: after the command flips, wait
    # until wheel speed drops below this before flipping the sign [m/s].
    reverse_speed_threshold = rx.Parameter(0.05)

    # -- IMU (direction estimation only) --
    imu_accel_sign = rx.Parameter(1.0)          # -1.0 if accel_x opposes base_link x
    imu_timeout = rx.Parameter(0.3)             # IMU considered stale after this [s]

    # ZUPT: wheel speed below this counts as standstill, which resets the
    # integrated speed and updates the bias estimate [m/s].
    zupt_speed_threshold = rx.Parameter(0.03)
    accel_bias_alpha = rx.Parameter(0.02)       # first-order LPF gain for the bias

    # The sign of the integrated speed is only trusted above this [m/s].
    imu_direction_threshold = rx.Parameter(0.12)

    # -- Sanity limits --
    max_dt = rx.Parameter(0.5)
    max_wheel_step = rx.Parameter(0.5)

    # -- Output --
    linear_covariance = rx.Parameter(0.05)
    angular_covariance = rx.Parameter(0.2)

    ## ---------------- State ---------------- ##

    _last_dist = None
    _last_time = None

    _dir = None
    _cmd_dir = None
    _rc_dir = None
    _rc_stamp = None

    _imu_t = None
    _imu_stamp = None
    _v_imu = 0.0
    _ax_bias = 0.0
    _stationary = True

    _warned_len = False

    ## ---------------- Publishers ---------------- ##

    twist_pub = rx.Publisher(TwistWithCovarianceStamped, twist_topic, qos_pubber)

    ## ---------------- Subscribers ---------------- ##

    @rx.Subscriber(ManualControl, control_topic, qos_subber)
    def control_cb(self, msg):
        if self.direction_source in ('rc', 'imu'):
            return
        delta = float(msg.z) - float(self.control_neutral)
        if delta < -self.control_deadband:
            self._set_cmd_dir(1)
        elif delta > self.control_deadband:
            self._set_cmd_dir(-1)
        # Inside the deadband, hold the previous direction.

    @rx.Subscriber(RCIn, rc_topic, qos_subber)
    def rc_cb(self, msg):
        if self.direction_source in ('manual_control', 'imu'):
            return

        i = int(self.rc_channel)
        if i >= len(msg.channels):
            return

        delta = int(msg.channels[i]) - int(self.rc_neutral)
        if self.rc_reversed:
            delta = -delta
        if abs(delta) <= self.rc_deadband:
            return

        self._rc_dir = 1 if delta > 0 else -1
        self._rc_stamp = self._now()

        if self.direction_source == 'rc':
            self._set_cmd_dir(self._rc_dir)

    @rx.Subscriber(Imu, imu_topic, qos_subber)
    def imu_cb(self, msg):
        """Integrate longitudinal acceleration with ZUPT.

        The result is only used to pick a sign; it never enters the twist.
        """

        t = Time.from_msg(msg.header.stamp).nanoseconds * 1.0e-9
        self._imu_stamp = self._now()

        ax = float(msg.linear_acceleration.x) * float(self.imu_accel_sign)

        if self._imu_t is None:
            self._imu_t = t
            return

        dt = t - self._imu_t
        self._imu_t = t

        if dt <= 0.0 or dt > self.max_dt:
            return

        if self._stationary:
            # Standstill: reset the speed and pull the bias towards the reading.
            self._v_imu = 0.0
            a = float(self.accel_bias_alpha)
            self._ax_bias += a * (ax - self._ax_bias)
        else:
            self._v_imu += (ax - self._ax_bias) * dt

    @rx.Subscriber(WheelOdomStamped, distance_topic, qos_subber)
    def distance_cb(self, msg):

        dist = list(msg.data)

        li = int(self.left_index)
        ri = int(self.right_index)

        if len(dist) <= max(li, ri):
            if not self._warned_len:
                self.get_logger().error(
                    f'Distance array has length {len(dist)}, cannot index {li}/{ri}. '
                    f'Check the count / send_raw settings of the mavros '
                    f'wheel_odometry plugin.'
                )
                self._warned_len = True
            return

        t = Time.from_msg(msg.header.stamp).nanoseconds * 1.0e-9

        if self._last_dist is None:
            self._last_dist = dist
            self._last_time = t
            self.get_logger().info('First wheel distance message received, baseline set')
            return

        dt = t - self._last_time
        d_l = dist[li] - self._last_dist[li]
        d_r = dist[ri] - self._last_dist[ri]

        self._last_dist = dist
        self._last_time = t

        if dt <= 0.0 or dt > self.max_dt:
            self.get_logger().warn(f'Invalid dt={dt:.4f}, skipping this frame')
            return

        # The counters are monotonic, so a negative or oversized increment means
        # they were reset.
        step = self.max_wheel_step
        if d_l < -1.0e-6 or d_r < -1.0e-6 or abs(d_l) > step or abs(d_r) > step:
            self.get_logger().warn(
                f'Implausible increment (dL={d_l:.4f}, dR={d_r:.4f}), '
                f'treating as a counter reset and skipping'
            )
            return

        # Unsigned wheel speed, used for the ZUPT check and the flip hysteresis.
        speed = abs(0.5 * (d_l + d_r)) / dt
        self._stationary = speed < self.zupt_speed_threshold

        # Restore the sign.
        sign = float(self._resolve_direction(speed))
        d_l *= sign
        d_r *= sign

        v = 0.5 * (d_l + d_r) / dt
        w = (d_r - d_l) / (self.axle_track * dt)

        self._publish(msg.header.stamp, v, w)

    ## ---------------- Helpers ---------------- ##

    def _now(self):
        return self.get_clock().now().nanoseconds * 1.0e-9

    def _imu_alive(self):
        return (self._imu_stamp is not None
                and self._now() - self._imu_stamp < self.imu_timeout)

    def _set_cmd_dir(self, d):
        self._cmd_dir = d
        if self._dir is None:
            self._dir = d

    def _imu_direction(self):
        """Sign of the integrated speed, or None when it is too small to trust."""
        if not self._imu_alive():
            return None
        if abs(self._v_imu) < self.imu_direction_threshold:
            return None
        return 1 if self._v_imu > 0.0 else -1

    def _command_direction(self):
        """Direction implied by the control command, honouring RC priority."""
        if self.direction_source == 'auto' and self._rc_stamp is not None:
            if self._now() - self._rc_stamp < self.rc_timeout and self._rc_dir is not None:
                return self._rc_dir
        return self._cmd_dir

    def _resolve_direction(self, speed):
        """Pick the sign to apply to this sample."""

        src = self.direction_source
        imu_dir = self._imu_direction() if src in ('imu', 'fusion') else None
        cmd_dir = self._command_direction()

        # The IMU observes actual motion, so no hysteresis is needed here.
        if imu_dir is not None:
            if src == 'imu' or (src == 'fusion' and cmd_dir != imu_dir):
                if self._dir != imu_dir:
                    self.get_logger().info(
                        f'Direction from IMU: {self._dir} -> {imu_dir:+d} '
                        f'(v_imu={self._v_imu:+.3f})'
                    )
                self._dir = imu_dir
                return self._dir

        if cmd_dir is None:
            return self._dir if self._dir is not None else int(self.default_direction)

        if self._dir is None:
            self._dir = cmd_dir
            return self._dir

        # Command-based: do not flip while coasting, wait until nearly stopped.
        if cmd_dir != self._dir and speed < self.reverse_speed_threshold:
            self.get_logger().info(
                f'Direction from command: {self._dir:+d} -> {cmd_dir:+d}'
            )
            self._dir = cmd_dir

        return self._dir

    def _publish(self, stamp, v, w):

        msg = TwistWithCovarianceStamped()
        msg.header.stamp = stamp
        msg.header.frame_id = self.base_frame

        msg.twist.twist.linear.x = v
        msg.twist.twist.angular.z = w

        msg.twist.covariance[0] = float(self.linear_covariance)    # vx
        msg.twist.covariance[7] = 1.0e6                            # vy
        msg.twist.covariance[14] = 1.0e6                           # vz
        msg.twist.covariance[21] = 1.0e6                           # roll rate
        msg.twist.covariance[28] = 1.0e6                           # pitch rate
        msg.twist.covariance[35] = float(self.angular_covariance)  # yaw rate

        self.twist_pub.publish(msg)

    ## ---------------- Lifecycle ---------------- ##

    def on_startup(self):

        valid = ('manual_control', 'rc', 'auto', 'imu', 'fusion')
        if self.direction_source not in valid:
            raise ValueError(
                f'direction_source must be one of {valid}, '
                f'got "{self.direction_source}"'
            )

        self._last_dist = None
        self._last_time = None
        self._dir = None
        self._cmd_dir = None

        self._imu_t = None
        self._imu_stamp = None
        self._v_imu = 0.0
        self._ax_bias = 0.0
        self._stationary = True

        self.get_logger().info(
            f'wheel_twist_node started\n'
            f'  distance  : {self.distance_topic}\n'
            f'  imu       : {self.imu_topic} (direction only)\n'
            f'  twist     : {self.twist_topic}\n'
            f'  direction : {self.direction_source}\n'
            f'  frame     : {self.base_frame}\n'
            f'  track     : {self.axle_track} m'
        )

    @rx.Timer(5.0)
    def watchdog(self):
        if self.direction_source in ('imu', 'fusion') and not self._imu_alive():
            self.get_logger().warn(
                f'No data on {self.imu_topic}, direction has fallen back to '
                f'the control command'
            )


if __name__ == '__main__':
    single_encoder_twist_filter.main()
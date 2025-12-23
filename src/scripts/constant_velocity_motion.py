#!/usr/bin/env python3
"""
Constant Velocity Motion Generator for 3-DOF Robot Arm

Sends constant velocity commands to all joints.
Velocity: 360 degrees / 10 seconds = 36 deg/s = 0.6283 rad/s

Publishes:
  - Velocity commands to Gazebo controller
  - Ground truth (commanded positions/velocities) to /ground_truth/joint_states
  - Motion start/stop signals

Author: Abithan
Date: December 2025
"""

import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray, Bool
from sensor_msgs.msg import JointState
import math


# Default parameters
# 360 degrees in 10 seconds = 2*pi / 10 = 0.6283 rad/s
DEFAULT_VELOCITY = 2.0 * math.pi / 10.0  # 0.6283 rad/s
DEFAULT_DURATION = 10.0  # seconds
DEFAULT_PUBLISH_RATE = 50.0  # Hz


class ConstantVelocityMotion(Node):
    """
    Generates constant velocity motion for all joints.
    
    All joints rotate at the same constant velocity:
    - Velocity: 360°/10s = 0.6283 rad/s
    - Duration: 10 seconds
    - Final position: 2π rad (360°)
    
    Ground Truth:
        θ(t) = ω * t  (linear position increase)
        θ̇(t) = ω     (constant velocity)
    """
    
    def __init__(self):
        super().__init__('constant_velocity_motion')
        self.get_logger().info("ConstantVelocityMotion initializing...")
        
    def init(self) -> bool:
        """Initialize the node."""
        # Declare parameters
        self.declare_parameter('velocity', DEFAULT_VELOCITY)
        self.declare_parameter('duration', DEFAULT_DURATION)
        self.declare_parameter('publish_rate', DEFAULT_PUBLISH_RATE)
        
        # Get parameters
        self.velocity = self.get_parameter('velocity').value
        self.duration = self.get_parameter('duration').value
        self.publish_rate = self.get_parameter('publish_rate').value
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("CONSTANT VELOCITY MOTION")
        self.get_logger().info("=" * 50)
        self.get_logger().info(f"Velocity: {self.velocity:.4f} rad/s ({math.degrees(self.velocity):.2f} deg/s)")
        self.get_logger().info(f"Duration: {self.duration} seconds")
        self.get_logger().info(f"Final Position: {self.velocity * self.duration:.4f} rad ({math.degrees(self.velocity * self.duration):.2f} deg)")
        self.get_logger().info("=" * 50)
        
        # Joint names
        self.joint_names = ['joint_1', 'joint_2', 'joint_3']
        self.n_joints = 3
        
        # Publishers
        # Velocity command to forward_velocity_controller
        self.pub_velocity_cmd = self.create_publisher(
            Float64MultiArray,
            '/forward_velocity_controller/commands',
            10
        )
        
        # Ground truth
        self.pub_ground_truth = self.create_publisher(
            JointState,
            '/ground_truth/joint_states',
            10
        )
        
        # Motion signals
        self.pub_motion_start = self.create_publisher(
            Bool, '/motion/start', 10)
        
        self.pub_motion_complete = self.create_publisher(
            Bool, '/motion/cycle_complete', 10)
        
        # State
        self.motion_started = False
        self.motion_complete = False
        self.start_time = None
        
        # Start motion after a delay
        self.create_timer(1.0, self.start_motion)
        
        self.get_logger().info("ConstantVelocityMotion initialized")
        return True
    
    def start_motion(self):
        """Start constant velocity motion."""
        if self.motion_started:
            return
        
        self.motion_started = True
        self.start_time = self.get_clock().now()
        
        self.get_logger().info("=== STARTING CONSTANT VELOCITY MOTION ===")
        self.get_logger().info(f"All joints: {self.velocity:.4f} rad/s")
        
        # Signal motion start
        start_msg = Bool()
        start_msg.data = True
        self.pub_motion_start.publish(start_msg)
        
        # Send velocity command
        vel_cmd = Float64MultiArray()
        vel_cmd.data = [self.velocity, self.velocity, self.velocity]
        self.pub_velocity_cmd.publish(vel_cmd)
        
        # Start ground truth publisher
        timer_period = 1.0 / self.publish_rate
        self.ground_truth_timer = self.create_timer(timer_period, self.publish_ground_truth)
        
        # Schedule motion stop
        self.create_timer(self.duration, self.stop_motion)
    
    def publish_ground_truth(self):
        """Publish ground truth = expected position and velocity."""
        if not self.motion_started or self.start_time is None:
            return
        
        # Calculate elapsed time
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        
        if self.motion_complete:
            # After motion stops, position stays constant, velocity = 0
            elapsed = self.duration
            velocities = [0.0, 0.0, 0.0]
        else:
            # During motion: constant velocity
            elapsed = min(elapsed, self.duration)
            velocities = [self.velocity] * self.n_joints
        
        # Ground truth position: θ(t) = ω * t
        positions = [self.velocity * elapsed] * self.n_joints
        
        # Publish
        msg = JointState()
        msg.header.stamp = current_time.to_msg()
        msg.name = self.joint_names
        msg.position = positions
        msg.velocity = velocities
        msg.effort = [0.0] * self.n_joints
        
        self.pub_ground_truth.publish(msg)
    
    def stop_motion(self):
        """Stop motion and signal completion."""
        if self.motion_complete:
            return
        
        self.motion_complete = True
        
        # Send zero velocity command
        vel_cmd = Float64MultiArray()
        vel_cmd.data = [0.0, 0.0, 0.0]
        self.pub_velocity_cmd.publish(vel_cmd)
        
        self.get_logger().info("=== MOTION COMPLETE ===")
        self.get_logger().info(f"Final position: {self.velocity * self.duration:.4f} rad")
        
        # Signal completion
        complete_msg = Bool()
        complete_msg.data = True
        self.pub_motion_complete.publish(complete_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = ConstantVelocityMotion()
    if not node.init():
        node.get_logger().error("Failed to initialize ConstantVelocityMotion")
        return
    
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()

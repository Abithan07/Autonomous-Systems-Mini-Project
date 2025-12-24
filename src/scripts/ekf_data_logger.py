#!/usr/bin/env python3
"""
Data Logger for EKF Velocity Model Evaluation

Logs:
    - Ground Truth: Commanded positions and velocities
    - Noisy Sensor: Gazebo reading + added noise
    - EKF Estimate: Filtered position and velocity

Author: Abithan
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, Bool
import numpy as np
import os
from datetime import datetime
import csv


DEFAULT_LOG_DIR = "/home/abithan_ubuntu/ros2_ws4/src/log_files"
DEFAULT_LOG_RATE = 50.0


class EKFDataLoggerVelocity(Node):
    """
    Data logger for velocity model EKF evaluation.
    
    Logs position AND velocity for all three signals.
    """
    
    def __init__(self):
        super().__init__('ekf_data_logger')
        self.get_logger().info("EKFDataLoggerVelocity initializing...")
        
    def init(self) -> bool:
        """Initialize the data logger."""
        # Parameters
        self.declare_parameter('log_directory', DEFAULT_LOG_DIR)
        self.declare_parameter('log_rate', DEFAULT_LOG_RATE)
        self.declare_parameter('start_immediately', False)  # Start logging without waiting for motion
        
        log_dir = self.get_parameter('log_directory').value
        self.log_rate = self.get_parameter('log_rate').value
        self.start_immediately = self.get_parameter('start_immediately').value
        
        # Create log directory
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        self.log_path = os.path.join(log_dir, f'ekf_velocity_{timestamp}')
        os.makedirs(self.log_path, exist_ok=True)
        
        self.get_logger().info(f"Log Directory: {self.log_path}")
        
        # Data storage
        self.ground_truth_pos = None
        self.ground_truth_vel = None
        self.noisy_pos = None
        self.noisy_vel = None
        self.ekf_pos = None
        self.ekf_vel = None
        self.covariance = None
        
        self.start_time = None
        self.logging_active = False
        self.motion_complete = False
        self.motion_started = False
        
        # Setup CSV
        csv_path = os.path.join(self.log_path, 'ekf_velocity_data.csv')
        self.csv_file = open(csv_path, 'w', newline='')
        self.csv_writer = csv.writer(self.csv_file)
        
        # Header - position only estimation (velocity used as control input)
        header = [
            'timestamp',
            # Ground truth positions
            'gt_pos1', 'gt_pos2', 'gt_pos3',
            # Ground truth velocities (for reference)
            'gt_vel1', 'gt_vel2', 'gt_vel3',
            # Noisy positions
            'noisy_pos1', 'noisy_pos2', 'noisy_pos3',
            # Sensor velocities (no noise)
            'sensor_vel1', 'sensor_vel2', 'sensor_vel3',
            # EKF estimated positions
            'ekf_pos1', 'ekf_pos2', 'ekf_pos3',
            # Covariance (position)
            'cov_pos1', 'cov_pos2', 'cov_pos3'
        ]
        self.csv_writer.writerow(header)
        
        # Subscribers
        self.sub_ground_truth = self.create_subscription(
            JointState, '/ground_truth/joint_states',
            self.ground_truth_callback, 10)
        
        self.sub_noisy = self.create_subscription(
            JointState, '/noisy/joint_states',
            self.noisy_callback, 10)
        
        self.sub_ekf = self.create_subscription(
            JointState, '/ekf/joint_states',
            self.ekf_callback, 10)
        
        self.sub_covariance = self.create_subscription(
            Float64MultiArray, '/ekf/covariance',
            self.covariance_callback, 10)
        
        self.sub_motion_start = self.create_subscription(
            Bool, '/motion/start',
            self.motion_start_callback, 10)
        
        self.sub_motion_complete = self.create_subscription(
            Bool, '/motion/cycle_complete',
            self.motion_complete_callback, 10)
        
        # Timer for logging
        timer_period = 1.0 / self.log_rate
        self.log_timer = self.create_timer(timer_period, self.log_data)
        
        self.data_count = 0
        
        if self.start_immediately:
            self.start_time = self.get_clock().now()
            self.logging_active = True
            self.motion_started = True  # Bypass motion start requirement
            self.get_logger().info("=== LOGGING STARTED IMMEDIATELY ===")
        else:
            self.get_logger().info("EKFDataLoggerVelocity initialized - waiting for motion start...")
        
        return True
    
    def motion_start_callback(self, msg: Bool):
        """Handle motion start signal."""
        if msg.data and not self.motion_started:
            self.motion_started = True
            self.start_time = self.get_clock().now()
            self.logging_active = True
            self.get_logger().info("=== MOTION START - LOGGING STARTED ===")
    
    def ground_truth_callback(self, msg: JointState):
        """Store ground truth."""
        self.ground_truth_pos = list(msg.position) if msg.position else [0.0, 0.0, 0.0]
        self.ground_truth_vel = list(msg.velocity) if msg.velocity else [0.0, 0.0, 0.0]
    
    def noisy_callback(self, msg: JointState):
        """Store noisy sensor reading."""
        self.noisy_pos = list(msg.position) if msg.position else [0.0, 0.0, 0.0]
        self.noisy_vel = list(msg.velocity) if msg.velocity else [0.0, 0.0, 0.0]
    
    def ekf_callback(self, msg: JointState):
        """Store EKF estimate."""
        self.ekf_pos = list(msg.position) if msg.position else [0.0, 0.0, 0.0]
        self.ekf_vel = list(msg.velocity) if msg.velocity else [0.0, 0.0, 0.0]
    
    def covariance_callback(self, msg: Float64MultiArray):
        """Store covariance."""
        self.covariance = list(msg.data) if msg.data else [0.0] * 6
    
    def motion_complete_callback(self, msg: Bool):
        """Handle motion completion."""
        if msg.data and not self.motion_complete:
            self.motion_complete = True
            self.get_logger().info("=== MOTION COMPLETE ===")
            self.create_timer(2.0, self.finalize_log)
    
    def log_data(self):
        """Log synchronized data to CSV."""
        if not self.logging_active:
            return
        
        # Debug: show what data we have
        if self.data_count == 0:
            self.get_logger().info(f"Data check - GT: {self.ground_truth_pos is not None}, "
                                   f"Noisy: {self.noisy_pos is not None}, "
                                   f"EKF: {self.ekf_pos is not None}")
        
        if (self.noisy_pos is None or 
            self.ekf_pos is None):
            return
        
        # Calculate timestamp
        current_time = self.get_clock().now()
        elapsed = (current_time - self.start_time).nanoseconds / 1e9
        
        # Build row
        row = [elapsed]
        
        # Ground truth (use zeros if not available yet - before motion starts)
        if self.ground_truth_pos is not None:
            row.extend(self.ground_truth_pos[:3])
            row.extend(self.ground_truth_vel[:3] if self.ground_truth_vel else [0.0, 0.0, 0.0])
        else:
            row.extend([0.0, 0.0, 0.0])  # No ground truth yet
            row.extend([0.0, 0.0, 0.0])
        
        # Noisy position
        row.extend(self.noisy_pos[:3])
        
        # Sensor velocity (no noise)
        row.extend(self.noisy_vel[:3] if self.noisy_vel else [0.0, 0.0, 0.0])
        
        # EKF estimated position
        row.extend(self.ekf_pos[:3])
        
        # Covariance (position)
        if self.covariance and len(self.covariance) >= 3:
            row.extend(self.covariance[:3])
        else:
            row.extend([0.0] * 3)
        
        self.csv_writer.writerow(row)
        self.data_count += 1
        
        if self.data_count % 100 == 0:
            self.get_logger().info(f"Logged {self.data_count} points | t={elapsed:.2f}s")
    
    def finalize_log(self):
        """Finalize and close log file."""
        self.logging_active = False
        self.csv_file.close()
        
        self.get_logger().info("=" * 60)
        self.get_logger().info("=== LOGGING COMPLETE ===")
        self.get_logger().info(f"Total data points: {self.data_count}")
        self.get_logger().info(f"Log saved to: {self.log_path}")
        self.get_logger().info("=" * 60)
        
        self.print_summary()
    
    def print_summary(self):
        """Print RMSE summary."""
        csv_path = os.path.join(self.log_path, 'ekf_velocity_data.csv')
        
        try:
            import pandas as pd
            df = pd.read_csv(csv_path)
            
            self.get_logger().info("\n=== RMSE ANALYSIS ===")
            self.get_logger().info("Model: Constant Velocity (Position Only Estimation)")
            self.get_logger().info("State: [θ1, θ2, θ3] (position only)\n")
            
            # Position RMSE
            self.get_logger().info("--- Position RMSE ---")
            for i in range(1, 4):
                noisy_err = df[f'noisy_pos{i}'] - df[f'gt_pos{i}']
                ekf_err = df[f'ekf_pos{i}'] - df[f'gt_pos{i}']
                noisy_rmse = np.sqrt(np.mean(noisy_err**2))
                ekf_rmse = np.sqrt(np.mean(ekf_err**2))
                improvement = (noisy_rmse - ekf_rmse) / noisy_rmse * 100
                self.get_logger().info(
                    f"Joint {i}: Noisy={noisy_rmse:.4f}, EKF={ekf_rmse:.4f}, Improvement={improvement:.1f}%"
                )
                
        except Exception as e:
            self.get_logger().warn(f"Could not compute summary: {e}")
    
    def destroy_node(self):
        """Clean up."""
        if hasattr(self, 'csv_file') and not self.csv_file.closed:
            self.csv_file.close()
        super().destroy_node()


def main(args=None):
    rclpy.init(args=args)
    
    node = EKFDataLoggerVelocity()
    if not node.init():
        node.get_logger().error("Failed to initialize EKFDataLoggerVelocity")
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

#!/usr/bin/env python3
"""
Extended Kalman Filter for 3-DOF Robot Arm - Position Only Estimation

Following lecture notation:
    Algorithm EKF(μ_{t-1}, Σ_{t-1}, u_t, z_t):
        μ̄_t = g(u_t, μ_{t-1})                           # Prediction (motion model)
        Σ̄_t = G_t Σ_{t-1} G_t^T + R_t                   # Predicted covariance
        K_t = Σ̄_t H_t^T (H_t Σ̄_t H_t^T + Q_t)^{-1}     # Kalman gain
        μ_t = μ̄_t + K_t(z_t - h(μ̄_t))                  # State update
        Σ_t = (I - K_t H_t) Σ̄_t                         # Covariance update
        return μ_t, Σ_t

Where:
    μ (mu)     = state mean [θ1, θ2, θ3]
    Σ (Sigma)  = state covariance
    G_t        = Jacobian of motion model g (∂g/∂μ)
    H_t        = Jacobian of observation model h (∂h/∂μ)
    R_t        = process/motion noise covariance
    Q_t        = measurement noise covariance
    K_t        = Kalman gain
    g()        = motion model: g(u_t, μ_{t-1}) = μ_{t-1} + u_t * Δt
    h()        = observation model: h(μ̄_t) = μ̄_t (direct observation)
    u_t        = control input (velocity from sensor)
    z_t        = measurement (noisy position)

Author: Abithan
Date: December 2025
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np


DEFAULT_PROCESS_NOISE = 0.01         # R_t: Process/motion noise (rad)
DEFAULT_MEASUREMENT_NOISE = 0.05     # Q_t: Measurement noise (rad)
DEFAULT_INITIAL_COV = 1.0
DEFAULT_EKF_RATE = 50.0              # Match with ground truth and logger rate


class ArmEKFPositionOnly(Node):
    """
    EKF Node - Position Only Estimation using Velocity Model.
    
    State: μ = [θ1, θ2, θ3] (position only)
    
    Motion Model g(u_t, μ_{t-1}):
        μ̄_t = μ_{t-1} + u_t * Δt
        where u_t = measured velocity (control input)
        
    Observation Model h(μ̄_t):
        z_t = h(μ̄_t) = μ̄_t  (direct position observation)
    
    Subscribes:
        /joint_states - Gazebo sensor (positions and velocities)
        
    Publishes:
        /noisy/joint_states - Position with noise, velocity clean
        /ekf/joint_states - EKF filtered position estimate
        /ekf/covariance - Current covariance diagonal
    """
    
    def __init__(self):
        super().__init__('arm_ekf_position_node')
        self.get_logger().info("ArmEKF (Position Only) initializing...")
        
    def init(self) -> bool:
        """Initialize EKF."""
        # Declare parameters
        self.declare_parameter('process_noise', DEFAULT_PROCESS_NOISE)
        self.declare_parameter('measurement_noise', DEFAULT_MEASUREMENT_NOISE)
        self.declare_parameter('initial_covariance', DEFAULT_INITIAL_COV)
        self.declare_parameter('ekf_rate', DEFAULT_EKF_RATE)
        
        # Get parameters
        self.process_noise = self.get_parameter('process_noise').value
        self.measurement_noise = self.get_parameter('measurement_noise').value
        self.initial_cov = self.get_parameter('initial_covariance').value
        self.ekf_rate = self.get_parameter('ekf_rate').value
        
        # Time step
        self.dt = 1.0 / self.ekf_rate
        
        self.get_logger().info("=" * 50)
        self.get_logger().info("EKF POSITION ONLY ESTIMATION")
        self.get_logger().info("=" * 50)
        self.get_logger().info("State: μ = [θ1, θ2, θ3] (position only)")
        self.get_logger().info("Motion Model: μ̄_t = μ_{t-1} + u_t * Δt")
        self.get_logger().info("Observation Model: z_t = h(μ̄_t) = μ̄_t")
        self.get_logger().info(f"Δt = {self.dt:.4f} s")
        self.get_logger().info(f"R_t (Process Noise): {self.process_noise} rad")
        self.get_logger().info(f"Q_t (Measurement Noise): {self.measurement_noise} rad")
        self.get_logger().info("=" * 50)
        
        # Number of joints/states
        self.n = 3  # [θ1, θ2, θ3]
        
        # Joint names
        self.joint_names = ['joint_1', 'joint_2', 'joint_3']
        
        # EKF State Variables
        # State mean μ = [θ1, θ2, θ3]
        self.mu = np.zeros(self.n)
        
        # State covariance Σ (3x3)
        self.Sigma = np.eye(self.n) * self.initial_cov
        
        # EKF Matrices (Lecture Notation)
        # G_t: Jacobian of motion model g w.r.t. state (∂g/∂μ)
        # Since g(u, μ) = μ + u*Δt, G = ∂g/∂μ = I
        self.G = np.eye(self.n)
        
        # R_t: Process/motion noise covariance
        self.R = np.eye(self.n) * (self.process_noise ** 2)
        
        # H_t: Jacobian of observation model h w.r.t. state (∂h/∂μ)
        # Since h(μ) = μ (direct observation), H = I
        self.H = np.eye(self.n)
        
        # Q_t: Measurement noise covariance
        self.Q = np.eye(self.n) * (self.measurement_noise ** 2)
        
        # Identity matrix
        self.I = np.eye(self.n)
        
        # State flags
        self.initialized = False
        self.gazebo_positions = None
        self.gazebo_velocities = None
        
        # Publishers
        self.pub_noisy = self.create_publisher(
            JointState, '/noisy/joint_states', 10)
        
        self.pub_ekf = self.create_publisher(
            JointState, '/ekf/joint_states', 10)
        
        self.pub_covariance = self.create_publisher(
            Float64MultiArray, '/ekf/covariance', 10)
        
        # Subscribers
        self.sub_gazebo = self.create_subscription(
            JointState, '/joint_states', self.gazebo_callback, 10)
        
        # Timer
        self.ekf_timer = self.create_timer(self.dt, self.ekf_cycle)
        
        self.get_logger().info("ArmEKF (Position Only) initialized - waiting for sensor data...")
        return True
    
    def gazebo_callback(self, msg: JointState):
        """
        Callback for Gazebo sensor reading.
        Extract positions and velocities.
        """
        positions = np.zeros(self.n)
        velocities = np.zeros(self.n)
        
        for i, name in enumerate(self.joint_names):
            if name in msg.name:
                idx = msg.name.index(name)
                if idx < len(msg.position):
                    positions[i] = msg.position[idx]
                if idx < len(msg.velocity):
                    velocities[i] = msg.velocity[idx]
        
        self.gazebo_positions = positions
        self.gazebo_velocities = velocities
    
    def add_measurement_noise(self, positions: np.ndarray) -> np.ndarray:
        """Add Gaussian noise to position readings (simulates sensor noise)."""
        noise = np.random.normal(0, self.measurement_noise, self.n)
        return positions + noise
    
    def motion_model_g(self, mu_prev: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Motion model: g(u_t, μ_{t-1})
        
        μ̄_t = μ_{t-1} + u_t * Δt
        
        Args:
            mu_prev: Previous state μ_{t-1}
            u: Control input u_t (velocity)
            
        Returns:
            μ̄_t: Predicted state
        """
        return mu_prev + u * self.dt
    
    def observation_model_h(self, mu_bar: np.ndarray) -> np.ndarray:
        """
        Observation model: h(μ̄_t)
        
        For direct position observation: h(μ̄_t) = μ̄_t
        
        Args:
            mu_bar: Predicted state μ̄_t
            
        Returns:
            Expected measurement
        """
        return mu_bar
    
    def ekf_predict(self, u: np.ndarray):
        # μ̄_t = g(u_t, μ_{t-1})
        mu_bar = self.motion_model_g(self.mu, u)
        
        # Σ̄_t = G_t Σ_{t-1} G_t^T + R_t
        Sigma_bar = self.G @ self.Sigma @ self.G.T + self.R
        
        return mu_bar, Sigma_bar
    
    def ekf_update(self, mu_bar: np.ndarray, Sigma_bar: np.ndarray, z: np.ndarray):
        # K_t = Σ̄_t H_t^T (H_t Σ̄_t H_t^T + Q_t)^{-1}
        S = self.H @ Sigma_bar @ self.H.T + self.Q
        K = Sigma_bar @ self.H.T @ np.linalg.inv(S)
        
        # μ_t = μ̄_t + K_t(z_t - h(μ̄_t))
        innovation = z - self.observation_model_h(mu_bar)
        self.mu = mu_bar + K @ innovation
        
        # Σ_t = (I - K_t H_t) Σ̄_t
        self.Sigma = (self.I - K @ self.H) @ Sigma_bar
    
    def ekf_cycle(self):
        if self.gazebo_positions is None or self.gazebo_velocities is None:
            return
        
        # Get control input u_t (velocity)
        u_t = self.gazebo_velocities.copy()
        
        # Get measurement z_t (noisy position)
        z_t = self.add_measurement_noise(self.gazebo_positions)
        
        # Initialize on first measurement
        if not self.initialized:
            self.mu = z_t.copy()
            self.Sigma = np.eye(self.n) * self.initial_cov
            self.initialized = True
            self.get_logger().info(f"EKF initialized with μ_0: {self.mu}")
            return
        
        # EKF Prediction
        mu_bar, Sigma_bar = self.ekf_predict(u_t)
        
        # EKF Update
        self.ekf_update(mu_bar, Sigma_bar, z_t)
        
        # Publish results
        self.publish_results(z_t, u_t)
    
    def publish_results(self, z_t: np.ndarray, u_t: np.ndarray):
        """Publish noisy measurement and EKF estimate."""
        stamp = self.get_clock().now().to_msg()
        
        # Publish noisy measurement z_t
        noisy_msg = JointState()
        noisy_msg.header.stamp = stamp
        noisy_msg.name = self.joint_names
        noisy_msg.position = z_t.tolist()
        noisy_msg.velocity = u_t.tolist()  # Velocity not noisy
        self.pub_noisy.publish(noisy_msg)
        
        # Publish EKF estimate μ_t
        ekf_msg = JointState()
        ekf_msg.header.stamp = stamp
        ekf_msg.name = self.joint_names
        ekf_msg.position = self.mu.tolist()
        ekf_msg.velocity = u_t.tolist()  # Pass through velocity
        self.pub_ekf.publish(ekf_msg)
        
        # Publish covariance diagonal (Σ_t)
        cov_msg = Float64MultiArray()
        cov_msg.data = np.diag(self.Sigma).tolist()
        self.pub_covariance.publish(cov_msg)


def main(args=None):
    rclpy.init(args=args)
    
    node = ArmEKFPositionOnly()
    if not node.init():
        node.get_logger().error("Failed to initialize ArmEKF (Position Only)")
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

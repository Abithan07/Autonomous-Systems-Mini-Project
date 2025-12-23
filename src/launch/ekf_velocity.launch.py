"""
EKF Velocity Model Evaluation Launch File

This launch file starts the velocity-based EKF implementation:
1. Robot simulation with Gazebo (velocity control)
2. Controllers (joint_state_broadcaster + forward_velocity_controller)
3. Constant velocity motion generator
4. Velocity model EKF estimator
5. Data logger

Model:
    θ_{k+1} = θ_k + θ̇_k * Δt
    θ̇_{k+1} = θ̇_k

Usage:
    ros2 launch 3_link_arm ekf_velocity.launch.py
"""

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.substitutions import LaunchConfiguration, Command
from launch_ros.actions import Node
from launch.launch_description_sources import PythonLaunchDescriptionSource
import os
from ament_index_python.packages import get_package_share_directory
import math


def generate_launch_description():
    # Get package directory
    pkg_name = '3_link_arm'
    pkg_share = get_package_share_directory(pkg_name)
    
    # Path to XACRO file
    xacro_file = os.path.join(pkg_share, 'description', 'robot.urdf.xacro')
    rviz_config = os.path.join(pkg_share, 'config', 'main.rviz')
    
    # Gazebo launch
    gazebo_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('gazebo_ros'),
                'launch',
                'gazebo.launch.py'
            )
        ),
        launch_arguments={}.items()
    )
    
    # Node to spawn the robot in Gazebo
    spawn_entity = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=[
            '-topic', 'robot_description',
            '-entity', '3_link_arm'
        ],
        output='screen'
    )
    
    # Velocity: 360 deg / 10 sec
    velocity_rad_s = 2.0 * math.pi / 10.0
    
    return LaunchDescription([
        # ==================== LAUNCH ARGUMENTS ====================
        DeclareLaunchArgument(
            'use_ros2_control',
            default_value='true',
            description='Use ROS2 control'
        ),
        DeclareLaunchArgument(
            'sim_mode',
            default_value='true',
            description='Enable simulation mode'
        ),
        
        # ==================== ROBOT STATE PUBLISHER ====================
        Node(
            package='robot_state_publisher',
            executable='robot_state_publisher',
            name='robot_state_publisher',
            output='screen',
            parameters=[{
                'robot_description': Command([
                    'xacro ', xacro_file,
                    ' use_ros2_control:=', LaunchConfiguration('use_ros2_control'),
                    ' sim_mode:=', LaunchConfiguration('sim_mode')
                ])
            }]
        ),
        
        # ==================== GAZEBO ====================
        gazebo_launch,
        
        # Wait for Gazebo to start, then spawn the robot
        TimerAction(
            period=5.0,
            actions=[spawn_entity]
        ),
        
        # ==================== CONTROLLERS ====================
        # Joint state broadcaster
        TimerAction(
            period=8.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['joint_state_broadcaster'],
                    output='screen',
                ),
            ]
        ),
        
        # Forward velocity controller (NOT trajectory controller)
        TimerAction(
            period=10.0,
            actions=[
                Node(
                    package='controller_manager',
                    executable='spawner',
                    arguments=['forward_velocity_controller'],
                    output='screen',
                ),
            ]
        ),
        
        # ==================== EKF ESTIMATOR (Position Only, uses velocity for prediction) ====================
        TimerAction(
            period=12.0,
            actions=[
                Node(
                    package='3_link_arm',
                    executable='ekf.py',
                    name='arm_ekf_position',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'ekf_rate': 50.0,             # Match with ground truth rate
                        # Process noise for position
                        'process_noise': 0.02,        # Position (rad)
                        # Measurement noise for position (increased)
                        'measurement_noise': 0.2,     # Position (rad) - increased from 0.05
                        # Initial covariance
                        'initial_covariance': 10.0,
                    }]
                ),
            ]
        ),
        
        # ==================== DATA LOGGER (start with EKF to capture initial covariance!) ====================
        TimerAction(
            period=12.0,  # Same time as EKF
            actions=[
                Node(
                    package='3_link_arm',
                    executable='ekf_data_logger.py',
                    name='ekf_data_logger',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        'log_directory': '/home/abithan_ubuntu/ros2_ws4/src/log_files',
                        'log_rate': 50.0,
                        'start_immediately': True,  # Start logging immediately to see initial Σ_0
                    }]
                ),
            ]
        ),
        
        # ==================== CONSTANT VELOCITY MOTION (after logger!) ====================
        TimerAction(
            period=13.0,
            actions=[
                Node(
                    package='3_link_arm',
                    executable='constant_velocity_motion.py',
                    name='constant_velocity_motion',
                    output='screen',
                    parameters=[{
                        'use_sim_time': True,
                        # 360 degrees in 10 seconds
                        'velocity': velocity_rad_s,
                        'duration': 10.0,
                        'publish_rate': 50.0,
                    }]
                ),
            ]
        ),
        
        # ==================== RVIZ ====================
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            output='screen',
            arguments=['-d', rviz_config]
        ),
    ])

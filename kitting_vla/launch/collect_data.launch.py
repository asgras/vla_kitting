"""
collect_data.launch.py — Data collection launch file.

Starts the full existing ROS2 stack (sim_full.launch.py) plus the
data_collector_node to record training data from the expert policy.

Usage:
  ros2 launch kitting_vla collect_data.launch.py
  ros2 launch kitting_vla collect_data.launch.py hardware_type:=isaac
"""
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, TimerAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    pkg_bringup = get_package_share_directory('kitting_bringup')

    hardware_type_arg = DeclareLaunchArgument(
        'hardware_type', default_value='mock',
        description='Hardware backend: mock | isaac')
    output_dir_arg = DeclareLaunchArgument(
        'output_dir', default_value='./data/raw',
        description='Directory to save episode HDF5 files')

    # Include the full existing stack (RSP + ros2_control + MoveIt + task planner)
    full_stack = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(pkg_bringup, 'launch', 'sim_full.launch.py')),
        launch_arguments={
            'hardware_type': LaunchConfiguration('hardware_type'),
            'use_rviz': 'false',
        }.items())

    # Data collector node (15s delay to let full stack start)
    data_collector = TimerAction(period=15.0, actions=[
        Node(
            package='kitting_vla',
            executable='data_collector_node',
            name='data_collector_node',
            output='screen',
            parameters=[{
                'output_dir': LaunchConfiguration('output_dir'),
                'record_hz': 10.0,
                'hardware_type': LaunchConfiguration('hardware_type'),
            }])])

    return LaunchDescription([
        hardware_type_arg,
        output_dir_arg,
        full_stack,
        data_collector,
    ])

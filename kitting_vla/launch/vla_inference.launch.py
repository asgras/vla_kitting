"""
vla_inference.launch.py — Launch VLA inference stack (no MoveIt).

Starts:
  - robot_state_publisher (URDF via xacro)
  - ros2_control_node
  - joint_state_broadcaster + arm + gripper controllers
  - vla_inference_node
  - (optional) joint_merger for Isaac Sim

Does NOT start: move_group, planning_scene_setup, perception, RViz.
"""
import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, ExecuteProcess, TimerAction
from launch.conditions import IfCondition
from launch.substitutions import Command, LaunchConfiguration, PythonExpression
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue


def generate_launch_description():
    pkg_desc = get_package_share_directory('kitting_description')
    pkg_moveit = get_package_share_directory('kitting_moveit_config')
    pkg_vla = get_package_share_directory('kitting_vla')

    xacro_file = os.path.join(pkg_desc, 'urdf', 'hc10dt_kitting.urdf.xacro')

    # ── Launch arguments ──
    hardware_type_arg = DeclareLaunchArgument(
        'hardware_type', default_value='mock',
        description='Hardware backend: mock | isaac')
    camera_mount_arg = DeclareLaunchArgument(
        'camera_mount', default_value='fixed',
        description='Camera mount type: none | wrist | fixed')
    checkpoint_arg = DeclareLaunchArgument(
        'checkpoint_path',
        default_value='./checkpoints/kitting_octo_small',
        description='Path to fine-tuned Octo model checkpoint')

    hardware_type = LaunchConfiguration('hardware_type')
    camera_mount = LaunchConfiguration('camera_mount')

    # ── URDF ──
    robot_description_content = ParameterValue(
        Command(['xacro ', xacro_file,
                 ' hardware_type:=', hardware_type,
                 ' camera_mount:=', camera_mount]),
        value_type=str)

    # ── robot_state_publisher ──
    rsp_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description_content}])

    # ── ros2_control_node ──
    controller_mgr_yaml = os.path.join(
        pkg_moveit, 'config', 'controller_manager.yaml')
    ros2_control_node = Node(
        package='controller_manager',
        executable='ros2_control_node',
        output='screen',
        parameters=[controller_mgr_yaml])

    # ── Controllers (3s delay) ──
    controllers_yaml = os.path.join(pkg_moveit, 'config', 'controllers.yaml')
    controllers_isaac_yaml = os.path.join(
        pkg_moveit, 'config', 'controllers_isaac.yaml')

    delayed_controllers = TimerAction(period=3.0, actions=[
        Node(
            package='controller_manager',
            executable='spawner',
            arguments=[
                'joint_state_broadcaster',
                'hc10dt_arm_controller',
                'gripper_controller',
                '--controller-manager', '/controller_manager',
                '--controller-manager-timeout', '60',
                '--activate-as-group',
                '--param-file', PythonExpression([
                    "'", controllers_isaac_yaml, "' if '", hardware_type,
                    "' == 'isaac' else '", controllers_yaml, "'"]),
            ],
            output='screen')])

    # ── Joint merger (Isaac Sim only) ──
    joint_merger_script = os.path.join(pkg_desc, 'scripts', 'joint_merger.py')
    joint_merger = ExecuteProcess(
        cmd=['python3', joint_merger_script],
        output='screen',
        condition=IfCondition(PythonExpression([
            "'", hardware_type, "' == 'isaac'"])))

    # ── VLA config ──
    vla_config = os.path.join(pkg_vla, 'config', 'vla_cell_config.yaml')

    # ── VLA inference node (10s delay to let controllers activate) ──
    vla_node = TimerAction(period=10.0, actions=[
        Node(
            package='kitting_vla',
            executable='vla_inference_node',
            name='vla_inference_node',
            output='screen',
            parameters=[{
                'cell_config': vla_config,
                'checkpoint_path': LaunchConfiguration('checkpoint_path'),
                'hardware_type': hardware_type,
            }])])

    return LaunchDescription([
        hardware_type_arg,
        camera_mount_arg,
        checkpoint_arg,
        rsp_node,
        ros2_control_node,
        delayed_controllers,
        joint_merger,
        vla_node,
    ])

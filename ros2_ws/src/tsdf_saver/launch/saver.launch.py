"""
=====================================================================
 * MIT License
 * 
 * Copyright (c) 2025 Omni Instrument Inc.
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in
 * all copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN
 * THE SOFTWARE.
 * ===================================================================== 
"""

import os

from launch import LaunchDescription
from launch.actions import (
    DeclareLaunchArgument,
    ExecuteProcess,
    RegisterEventHandler,
    SetEnvironmentVariable,
    TimerAction,
    Shutdown
)
from launch.event_handlers import OnProcessExit
from launch.substitutions import EnvironmentVariable, LaunchConfiguration

from launch_ros.actions import ComposableNodeContainer, Node
from launch_ros.descriptions import ComposableNode


def generate_launch_description():

    home = os.path.expanduser("~")
    default_config_path = os.path.join(home, "ros2_ws", "src", "tsdf_saver", "config", "custom.yaml")
    default_bag_path = os.path.join(home, "dataset", "VIO_stripped")
    qos_yaml = os.path.join(home, "dataset", "qos_override.yaml")
    venv_site_packages = "/opt/venv/lib/python3.12/site-packages"


    bag_path = LaunchConfiguration("bag")
    save_threshold = LaunchConfiguration("save_time_threshold")
    config_file_path = LaunchConfiguration("dbtsdf_config")

    # ==============================================================
    #  Stereo Depth (StereoSGBM + WLS) — CLASSICAL
    # ==============================================================
    stereo_depth_node = Node(
        package='stereo_depth',
        executable='stereo_depth_node',
        name='stereo_depth_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'min_depth': 0.1},
            {'max_depth': 10.0},
            {'num_disparities': 128},
            {'block_size': 5},
            {'wls_lambda': 8000.0},
            {'wls_sigma': 1.5},
        ],
    )

    # ==============================================================
    #  Neural Depth (RAFT-Stereo + CUDA)
    # ==============================================================

    raft_model_path = os.path.join(
        home, 'ros2_ws', 'src', 'RAFT-Stereo', 'models', 'raftstereo-middlebury.pth'
    )

    neural_depth_node = Node(
        package='neural_depth',
        executable='neural_depth_node',
        name='neural_depth_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'model_path': raft_model_path},
            {'iters': 24},
            {'mixed_precision': True},
            {'corr_implementation': 'alt'},
            {'n_downsample': 2},
            {'context_norm': 'batch'},
            {'slow_fast_gru': False},
            {'n_gru_layers': 3},
            {'hidden_dims': [128, 128, 128]},
            {'corr_levels': 4},
            {'corr_radius': 4},
            {'shared_backbone': False},
            {'min_depth': 0.1},
            {'max_depth': 10.0},
        ],
    )

    # ==============================================================
    #  Neural Depth (HITNET ONNX) — commented out
    # ==============================================================
    hitnet_model_path = os.path.join(
        home, 'ros2_ws', 'src', 'neural_depth', 'models', 'hitnet', 'model_float32.onnx'
    )
    hitnet_depth_node = Node(
        package='neural_depth',
        executable='hitnet_node',
        name='hitnet_depth_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            {'model_path': hitnet_model_path},
            {'min_depth': 0.1},
            {'max_depth': 10.0},
        ],
    )

    # ==============================================================
    #  PointCloud Node (XYZ)
    # ==============================================================

    pointcloud_component = ComposableNode(
        package="depth_image_proc",
        plugin="depth_image_proc::PointCloudXyzNode",
        name="depth_image_pointcloud",
        parameters=[
            {"use_sim_time": True},
            {"queue_size": 20}
        ],
        remappings=[
            # image_transport derives camera_info automatically as:
            #   parent_namespace(/omni_depth/depth_registered) + /camera_info
            #   = /omni_depth/camera_info
            # The active depth node publishes there directly, avoiding clash with the bag.
            ("image_rect", "/omni_depth/depth_registered"),
            ("points",     "/stereo/points")
        ],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==============================================================
    #  TSDF Saver Component
    # ==============================================================

    tsdf_saver_component = ComposableNode(
        package="tsdf_saver",
        plugin="tsdf_saver::ExactTimeSaver",
        name="tsdf_saver",
        parameters=[
            {"use_sim_time": True},
            {"save_time_threshold": save_threshold}
        ],
        remappings=[
            ("cloud_in", "/stereo/points")
        ],
        extra_arguments=[{'use_intra_process_comms': True}]
    )

    # ==============================================================
    #  DB-TSDF Node (Normal ROS2 Node)
    # ==============================================================

    db_tsdf_node = Node(
        package='db_tsdf',
        executable='db_tsdf_node',
        name='db_tsdf_node',
        output='screen',
        parameters=[
            {'use_sim_time': True},
            config_file_path
        ],
        remappings=[
            ("cloud", "/tsdf/local_cloud")
        ]
    )

    # ==============================================================
    #  Container
    # ==============================================================

    container = ComposableNodeContainer(
        name="stereo_tsdf_container",
        namespace="",
        package="rclcpp_components",
        executable="component_container_mt",
        arguments=['--use_multi_threaded_executor', '--ros-args', '--log-level', 'info'],
        output="screen",
        parameters=[{"use_sim_time": True}],
        composable_node_descriptions=[
            pointcloud_component,
            tsdf_saver_component,
        ]
    )

    # ==============================================================
    #  Bag Playback Process
    # ==============================================================

    bag_proc = ExecuteProcess(
        cmd=[
            "ros2", "bag", "play",
            bag_path,
            "--clock",
            "--qos-profile-overrides-path", qos_yaml
        ],
        output="screen"
    )

    # ==============================================================
    #  Shutdown logic: 20 seconds after bag finishes
    # ==============================================================

    delayed_shutdown = RegisterEventHandler(
        OnProcessExit(
            target_action=bag_proc,
            on_exit=[
                TimerAction(
                    period=20.0,
                    actions=[
                        Shutdown(reason="Bag finished; shutting down after 20 seconds")
                    ]
                )
            ]
        )
    )

    # ==============================================================
    #  Launch Description
    # ==============================================================

    return LaunchDescription([
        SetEnvironmentVariable(
            "PYTHONPATH",
            [venv_site_packages, ":", EnvironmentVariable("PYTHONPATH", default_value="")]
        ),

        DeclareLaunchArgument(
            "bag",
            default_value=default_bag_path,
            description="Path to rosbag directory"
        ),

        DeclareLaunchArgument(
            "save_time_threshold",
            default_value="1765058989.0",
            description="When pointcloud time exceeds this value, TSDF saver triggers"
        ),

        DeclareLaunchArgument(
            "dbtsdf_config",
            default_value=default_config_path,
            description="Path to DB-TSDF config directory"
        ),

        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', os.path.join(home, 'ros2_ws', 'src', 'tsdf_saver', 'config', 'tsdf.rviz')],
            parameters=[{'use_sim_time': True}],
            output='screen'
        ),

        bag_proc,
        hitnet_depth_node,
        container,
        db_tsdf_node,
        delayed_shutdown
    ])

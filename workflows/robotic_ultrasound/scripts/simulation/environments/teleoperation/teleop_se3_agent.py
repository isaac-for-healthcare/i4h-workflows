# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

# Copyright (c) 2022-2024, The Isaac Lab Project Developers.
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause


import argparse
import os

from omni.isaac.lab.app import AppLauncher

# add argparse arguments
parser = argparse.ArgumentParser(description="Keyboard teleoperation for Isaac Lab environments.")
parser.add_argument(
    "--disable_fabric",
    action="store_true",
    default=False,
    help="Disable fabric and use USD I/O operations.",
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
parser.add_argument(
    "--teleop_device",
    type=str,
    default="keyboard",
    help="Device for interacting with environment",
)
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0",
    help="Name of the task. Default is Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0.",
)
parser.add_argument("--sensitivity", type=float, default=1.0, help="Sensitivity factor.")
parser.add_argument(
    "--topic_in_probe_pos",
    type=str,
    default="topic_ultrasound_info",
    help="topic name to consume probe pos",
)

parser.add_argument(
    "--rti_license_file",
    type=str,
    default=None,
    help="the path of rti_license_file. Default will use environment variables `RTI_LICENSE_FILE`",
)
parser.add_argument(
    "--viz_domain_id",
    type=int,
    default=0,
    help="domain id to publish data for visualization.",
)


# append AppLauncher cli args
AppLauncher.add_app_launcher_args(parser)

# Add after existing parser arguments
parser.add_argument(
    "--topic_in_room_camera",
    type=str,
    default="topic_room_camera_data_rgb",
    help="topic name to consume room camera rgb",
)
parser.add_argument(
    "--topic_in_room_camera_depth",
    type=str,
    default="topic_room_camera_data_depth",
    help="topic name to consume room camera depth",
)
parser.add_argument(
    "--topic_in_wrist_camera",
    type=str,
    default="topic_wrist_camera_data_rgb",
    help="topic name to consume wrist camera rgb",
)
parser.add_argument(
    "--topic_in_wrist_camera_depth",
    type=str,
    default="topic_wrist_camera_data_depth",
    help="topic name to consume wrist camera depth",
)

# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

"""Rest everything follows."""


import time  # noqa: F401, E402

import gymnasium as gym  # noqa: F401, E402
import omni.isaac.lab.utils.math as math_utils  # noqa: F401, E402
import omni.isaac.lab_tasks  # noqa: F401, E402
import omni.log  # noqa: F401, E402
import torch  # noqa: F401, E402
from dds.publisher import Publisher  # noqa: F401, E402
from dds.schemas.camera_info import CameraInfo  # noqa: F401, E402
from dds.schemas.franka_info import FrankaInfo  # noqa: F401, E402
from dds.schemas.usp_info import UltraSoundProbeInfo  # noqa: F401, E402
from omni.isaac.lab.devices import Se3Keyboard, Se3SpaceMouse  # noqa: F401, E402
from omni.isaac.lab.managers import SceneEntityCfg  # noqa: F401, E402
from omni.isaac.lab.managers import TerminationTermCfg as DoneTerm  # noqa: F401, E402
from omni.isaac.lab_tasks.manager_based.manipulation.lift import mdp  # noqa: F401, E402
from omni.isaac.lab_tasks.utils import parse_env_cfg  # noqa: F401, E402
# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401, E402
from simulation.environments.state_machine.utils import (
    capture_camera_images,
    compute_transform_sequence,
    get_joint_states,
    get_probe_pos_ori,
)

# Add RTI DDS imports
if args_cli.rti_license_file is not None:
    if not os.path.isabs(args_cli.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file


# Add publisher classes before main()
pub_data = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,  # Added for robot pose
    "room_cam_depth": None,
    "wrist_cam_depth": None,
}
hz = 30


class RoomCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_room_camera, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = 224
        output.width = 224
        output.data = pub_data["room_cam"].tobytes()
        return output


class RoomCamDepthPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_room_camera_depth, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = 224
        output.width = 224
        output.data = pub_data["room_cam_depth"].tobytes()
        return output


class WristCamPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_wrist_camera, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.height = 224
        output.width = 224
        output.data = pub_data["wrist_cam"].tobytes()
        return output


class WristCamDepthPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_wrist_camera_depth, CameraInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.height = 224
        output.width = 224
        output.data = pub_data["wrist_cam_depth"].tobytes()
        return output


# Add new publisher class with existing publishers
class PosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__("topic_franka_info", FrankaInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = FrankaInfo()
        output.joints_state_positions = pub_data["joint_pos"].tolist()
        return output


class ProbePosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_probe_pos, UltraSoundProbeInfo, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = UltraSoundProbeInfo()
        output.position = pub_data["probe_pos"].tolist()
        output.orientation = pub_data["probe_ori"].tolist()
        return output


def main():
    """Running keyboard teleoperation with Isaac Lab manipulation environment."""
    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task,
        device=args_cli.device,
        num_envs=args_cli.num_envs,
        use_fabric=not args_cli.disable_fabric,
    )
    # modify configuration
    env_cfg.terminations.time_out = None
    if "Lift" in args_cli.task:
        # set the resampling time range to large number to avoid resampling
        env_cfg.commands.object_pose.resampling_time_range = (1.0e9, 1.0e9)
        # add termination condition for reaching the goal otherwise the environment won't reset
        env_cfg.terminations.object_reached_goal = DoneTerm(func=mdp.object_reached_goal)
    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    # check environment name (for reach , we don't allow the gripper)
    if "Reach" in args_cli.task:
        omni.log.warn(
            f"The environment '{args_cli.task}' does not support gripper control. The device command will be ignored."
        )

    # create controller
    if args_cli.teleop_device.lower() == "keyboard":
        teleop_interface = Se3Keyboard(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.15 * args_cli.sensitivity,
        )
    elif args_cli.teleop_device.lower() == "spacemouse":
        teleop_interface = Se3SpaceMouse(
            pos_sensitivity=0.05 * args_cli.sensitivity,
            rot_sensitivity=0.015 * args_cli.sensitivity,
        )
    else:
        raise ValueError(f"Invalid device interface '{args_cli.teleop_device}'. Supported: 'keyboard', 'spacemouse'.")
    # add teleoperation key for env reset
    teleop_interface.add_callback("L", env.reset)
    # print helper for keyboard
    print(teleop_interface)

    # reset environment
    env.reset()
    teleop_interface.reset()

    # Add rate limiting variables
    target_hz = 30.0
    target_dt = 1.0 / target_hz
    last_step_time = time.time()

    # Create publishers for cameras
    viz_r_cam_writer = RoomCamPublisher(args_cli.viz_domain_id)
    viz_w_cam_writer = WristCamPublisher(args_cli.viz_domain_id)
    # viz_pos_writer = PosPublisher(args_cli.viz_domain_id)
    viz_probe_pos_writer = ProbePosPublisher(args_cli.viz_domain_id)
    viz_r_cam_depth_writer = RoomCamDepthPublisher(args_cli.viz_domain_id)
    viz_w_cam_depth_writer = WristCamDepthPublisher(args_cli.viz_domain_id)
    # Added robot pose publisher

    # simulate environment
    while simulation_app.is_running():
        current_time = time.time()
        dt = current_time - last_step_time
        if dt < target_dt:
            time.sleep(target_dt - dt)
            continue
        last_step_time = current_time

        with torch.inference_mode():
            delta_pose, gripper_command = teleop_interface.advance()
            delta_pose = torch.tensor(delta_pose.astype("float32"), device=env.unwrapped.device).repeat(
                env.unwrapped.num_envs, 1
            )

            delta_pos = delta_pose[:, :3]
            delta_rot = math_utils.quat_from_euler_xyz(delta_pose[:, 3], delta_pose[:, 4], delta_pose[:, 5])

            # get the robot's TCP pose
            robot_entity_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["panda_hand"])
            robot_entity_cfg.resolve(env.unwrapped.scene)
            ee_pose_w = env.unwrapped.scene["robot"].data.body_state_w[:, robot_entity_cfg.body_ids[0], 0:7].clone()
            ee_pos_w = ee_pose_w[:, :3]
            ee_rot_w = ee_pose_w[:, 3:7]

            # compute the target pose in the world frame
            target_pos_w, target_rot_w = math_utils.combine_frame_transforms(ee_pos_w, ee_rot_w, delta_pos, delta_rot)

            # compute the error in the pose in the world frame
            delta_pos, delta_rot = math_utils.compute_pose_error(
                ee_pos_w, ee_rot_w, target_pos_w, target_rot_w, rot_error_type="quat"
            )
            delta_rot[delta_rot.abs() < 1e-6] = 0.0

            # convert to angle-axis because it's needed for the action space
            target_rot_angle_axis = math_utils.axis_angle_from_quat(delta_rot)
            actions = torch.cat([delta_pos, target_rot_angle_axis], dim=1)

            env.step(actions)

            # Get the pose of the mesh objects (mesh)
            # The pose of the mesh objects are aligned with the organ (organ) in the US image view (us)
            # The US is attached to the end-effector (ee), so we have the following computation logics:
            # Each frame-to-frame transformation is available in the scene
            # mesh -> organ -> ee -> us
            quat_mesh_to_us, pos_mesh_to_us = compute_transform_sequence(env, ["mesh", "organ", "ee", "us"])
            pub_data["probe_pos"], pub_data["probe_ori"] = get_probe_pos_ori(quat_mesh_to_us, pos_mesh_to_us)

            # Get and publish camera images
            rgb_images, depth_images = capture_camera_images(
                env, ["room_camera", "wrist_camera"], device=env.unwrapped.device
            )
            pub_data["room_cam"] = rgb_images[0, 0, ...].cpu().numpy()
            pub_data["room_cam_depth"] = depth_images[0, 0, ...].cpu().numpy()
            pub_data["wrist_cam"] = rgb_images[0, 1, ...].cpu().numpy()
            pub_data["wrist_cam_depth"] = depth_images[0, 1, ...].cpu().numpy()
            pub_data["joint_pos"] = get_joint_states(env)[0]

            viz_r_cam_writer.write()
            viz_w_cam_writer.write()
            viz_probe_pos_writer.write()
            viz_r_cam_depth_writer.write()
            viz_w_cam_depth_writer.write()
    env.close()


if __name__ == "__main__":
    # run the main function
    # rti.asyncio.run(main())
    main()
    # close sim app
    simulation_app.close()

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

import argparse
import collections
import os
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.franka_ctrl import FrankaCtrlInput
from dds.schemas.franka_info import FrankaInfo
from dds.schemas.usp_info import UltraSoundProbeInfo
from dds.subscriber import SubscriberWithQueue
from isaaclab.app import AppLauncher
from simulation.environments.state_machine.utils import (
    RobotPositions,
    RobotQuaternions,
    capture_camera_images,
    compute_relative_action,
    compute_transform_sequence,
    get_joint_states,
    get_probe_pos_ori,
    get_robot_obs,
    reset_scene_to_initial_state,
    validate_hdf5_path,
)

# add argparse arguments
parser = argparse.ArgumentParser(description="Run simulation in a single-arm manipulator, communication via DDS.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-Teleop-Torso-FrankaUsRs-IK-RL-Rel-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--rti_license_file",
    type=str,
    default=os.getenv("RTI_LICENSE_FILE"),
    help="the path of rti_license_file. Default will use environment variables `RTI_LICENSE_FILE`",
)
parser.add_argument("--infer_domain_id", type=int, default=0, help="domain id to publish data for inference.")
parser.add_argument("--viz_domain_id", type=int, default=1, help="domain id to publish data for visualization.")
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
parser.add_argument(
    "--topic_in_franka_pos",
    type=str,
    default="topic_franka_info",
    help="topic name to consume franka pos",
)
parser.add_argument(
    "--topic_in_probe_pos",
    type=str,
    default="topic_ultrasound_info",
    help="topic name to consume probe pos",
)
parser.add_argument(
    "--topic_out",
    type=str,
    default="topic_franka_ctrl",
    help="topic name to publish generated franka actions",
)
parser.add_argument("--log_probe_pos", action="store_true", default=False, help="Log probe position.")
parser.add_argument(
    "--scale", type=float, default=1000.0, help="Scale factor to convert from omniverse to organ coordinate system."
)
parser.add_argument(
    "--hdf5_path",
    type=str,
    default=None,
    help="Path to single .hdf5 file or directory containing recorded data for environment reset.",
)
parser.add_argument(
    "--npz_prefix",
    type=str,
    default="",
    help="prefix to save the end-effector trajectory data during evaluation, only used when hdf5_path is provided.",
)

# append AppLauncher cli argruments
AppLauncher.add_app_launcher_args(parser)
# parse the arguments
args_cli = parser.parse_args()

# launch omniverse app
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
reset_flag = False

# isort: off
from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

# Import extensions to set up environment tasks
from robotic_us_ext import tasks  # noqa: F401

# isort: on

pub_data = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
    "probe_pos": None,
    "probe_ori": None,
    "room_cam_depth": None,
    "wrist_cam_depth": None,
}
hz = 30


class RoomCamPublisher(Publisher):
    def __init__(self, topic: str, domain_id: int, rgb: bool = True):
        super().__init__(topic, CameraInfo, 1 / hz, domain_id)
        self.rgb = rgb

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = 224
        output.width = 224
        if self.rgb:
            output.data = pub_data["room_cam"].tobytes()
        else:
            output.data = pub_data["room_cam_depth"].tobytes()
        return output


class WristCamPublisher(Publisher):
    def __init__(self, topic: str, domain_id: int, rgb: bool = True):
        super().__init__(topic, CameraInfo, 1 / hz, domain_id)
        self.rgb = rgb

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.height = 224
        output.width = 224
        if self.rgb:
            output.data = pub_data["wrist_cam"].tobytes()
        else:
            output.data = pub_data["wrist_cam_depth"].tobytes()
        return output


class PosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_franka_pos, FrankaInfo, 1 / hz, domain_id)

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


def get_reset_action(env, use_rel: bool = True):
    """Get the reset action."""
    reset_pos = torch.tensor(RobotPositions.SETUP, device=args_cli.device)
    reset_quat = torch.tensor(RobotQuaternions.DOWN, device=args_cli.device)
    reset_tensor = torch.cat([reset_pos, reset_quat], dim=-1)
    reset_tensor = reset_tensor.repeat(env.unwrapped.num_envs, 1)
    if not use_rel:
        return reset_tensor
    else:
        robot_obs = get_robot_obs(env)
        return compute_relative_action(reset_tensor, robot_obs)


def main():
    """Main function."""

    # parse configuration
    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # modify configuration
    env_cfg.terminations.time_out = None

    # create environment
    env = gym.make(args_cli.task, cfg=env_cfg)

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    # reset environment
    obs = env.reset()

    if args_cli.rti_license_file is not None:
        if not os.path.isabs(args_cli.rti_license_file):
            raise ValueError("RTI license file must be an existing absolute path.")
        os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file

    max_timesteps = 250
    if "Rel" in args_cli.task:
        action_dim = 6
    else:
        action_dim = 7
    if args_cli.hdf5_path is not None:
        reset_to_recorded_data = True
        episode_idx = 0
        torso_obs_key = "observations/torso_obs"
        joint_state_key = "abs_joint_pos"
        joint_vel_key = "observations/joint_vel"
        if "Rel" in args_cli.task:
            action_key = "action"
        else:
            action_key = "abs_action"
    else:
        reset_to_recorded_data = False
        # Recommended to use 40 steps to allow enough steps to reset the SETUP position of the robot
        reset_steps = 40

        # allow environment to settle
        for _ in range(reset_steps):
            reset_tensor = get_reset_action(env)
            obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    infer_r_cam_writer = RoomCamPublisher(topic=args_cli.topic_in_room_camera, domain_id=args_cli.infer_domain_id)
    infer_w_cam_writer = WristCamPublisher(topic=args_cli.topic_in_wrist_camera, domain_id=args_cli.infer_domain_id)
    infer_pos_writer = PosPublisher(args_cli.infer_domain_id)
    viz_r_cam_writer = RoomCamPublisher(topic=args_cli.topic_in_room_camera, domain_id=args_cli.viz_domain_id)
    viz_w_cam_writer = WristCamPublisher(topic=args_cli.topic_in_wrist_camera, domain_id=args_cli.viz_domain_id)
    viz_r_cam_depth_writer = RoomCamPublisher(
        topic=args_cli.topic_in_room_camera_depth, domain_id=args_cli.viz_domain_id, rgb=False
    )
    viz_w_cam_depth_writer = WristCamPublisher(
        topic=args_cli.topic_in_wrist_camera_depth, domain_id=args_cli.viz_domain_id, rgb=False
    )
    viz_pos_writer = PosPublisher(args_cli.viz_domain_id)
    viz_probe_pos_writer = ProbePosPublisher(args_cli.viz_domain_id)
    infer_reader = SubscriberWithQueue(args_cli.infer_domain_id, args_cli.topic_out, FrankaCtrlInput, 1 / hz)
    infer_reader.start()

    # Number of steps played before replanning
    replan_steps = 5

    if reset_to_recorded_data:
        total_episodes = validate_hdf5_path(args_cli.hdf5_path)
        print(f"total_episodes: {total_episodes}")
    else:
        total_episodes = 1

    # simulate environment
    while simulation_app.is_running():
        global reset_flag
        with torch.inference_mode():
            for episode_idx in range(total_episodes):
                print(f"\nepisode_idx: {episode_idx}")
                if reset_to_recorded_data:
                    actions = reset_scene_to_initial_state(
                        env,
                        args_cli.hdf5_path,
                        episode_idx,
                        action_key,
                        torso_obs_key,
                        joint_state_key,
                        joint_vel_key,
                    )
                    first_action = torch.tensor(actions[1], device=args_cli.device)
                    first_action = first_action.unsqueeze(0)
                    env.step(first_action)
                    print(f"Reset to recorded data: {get_robot_obs(env)}")
                action_plan = collections.deque()
                robot_obs = []
                for t in range(max_timesteps):
                    # get and publish the current images and joint positions
                    rgb_images, depth_images, _ = capture_camera_images(
                        env, ["room_camera", "wrist_camera"], device=env.unwrapped.device
                    )
                    pub_data["room_cam"], pub_data["room_cam_depth"] = (
                        rgb_images[0, 0, ...].cpu().numpy(),
                        depth_images[0, 0, ...].cpu().numpy(),
                    )
                    pub_data["wrist_cam"], pub_data["wrist_cam_depth"] = (
                        rgb_images[0, 1, ...].cpu().numpy(),
                        depth_images[0, 1, ...].cpu().numpy(),
                    )
                    pub_data["joint_pos"] = get_joint_states(env)[0]
                    robot_obs.append(get_robot_obs(env))

                    # Get the pose of the mesh objects (mesh)
                    # The mesh objects are aligned with the organ (organ) in the US image view (us)
                    # The US is attached to the end-effector (ee), so we have the following computation logics:
                    # Each frame-to-frame transformation is available in the scene
                    # mesh -> organ -> ee -> us
                    quat_mesh_to_us, pos_mesh_to_us = compute_transform_sequence(env, ["mesh", "organ", "ee", "us"])
                    pub_data["probe_pos"], pub_data["probe_ori"] = get_probe_pos_ori(
                        quat_mesh_to_us, pos_mesh_to_us, scale=args_cli.scale, log=args_cli.log_probe_pos
                    )
                    viz_r_cam_writer.write()
                    viz_w_cam_writer.write()
                    viz_r_cam_depth_writer.write()
                    viz_w_cam_depth_writer.write()
                    viz_pos_writer.write()
                    viz_probe_pos_writer.write()
                    if not action_plan:
                        # publish the images and joint positions when run policy inference
                        infer_r_cam_writer.write()
                        infer_w_cam_writer.write()
                        infer_pos_writer.write()

                        ret = None
                        while ret is None:
                            ret = infer_reader.read_data()
                            # Re-publish camera data while waiting
                            infer_r_cam_writer.write()
                            infer_w_cam_writer.write()
                            infer_pos_writer.write()
                        o: FrankaCtrlInput = ret
                        action_chunk = np.array(o.joint_positions, dtype=np.float32).reshape(-1, action_dim)
                        action_plan.extend(action_chunk[:replan_steps])

                    action = action_plan.popleft()

                    action = action.astype(np.float32)

                    # convert to torch
                    action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                    # step the environment
                    obs, rew, terminated, truncated, info_ = env.step(action)

                env.reset()
                if reset_to_recorded_data:
                    robot_obs = torch.stack(robot_obs, dim=0)
                    print(
                        f"robot_obs shape: {robot_obs.shape}, saved to {args_cli.hdf5_path}/robot_obs_{episode_idx}.npz"
                    )
                    if args_cli.hdf5_path.endswith(".hdf5"):
                        save_path = os.path.join(
                            Path(args_cli.hdf5_path).parent, f"{args_cli.npz_prefix}_robot_obs_{episode_idx}.npz"
                        )
                    else:
                        save_path = os.path.join(
                            args_cli.hdf5_path, f"{args_cli.npz_prefix}_robot_obs_{episode_idx}.npz"
                        )
                    save_dir = os.path.dirname(save_path)
                    os.makedirs(save_dir, exist_ok=True)
                    np.savez(save_path, robot_obs=robot_obs.cpu().numpy())

                    if episode_idx + 1 >= total_episodes:
                        print(f"Completed all episodes ({total_episodes})")
                        break

                    actions = reset_scene_to_initial_state(
                        env,
                        args_cli.hdf5_path,
                        episode_idx + 1,
                        action_key,
                        torso_obs_key,
                        joint_state_key,
                        joint_vel_key,
                    )
                    first_action = torch.tensor(actions[1], device=args_cli.device)
                    first_action = first_action.unsqueeze(0)
                    obs, rew, terminated, truncated, info_ = env.step(first_action)
                else:
                    for _ in range(reset_steps):
                        reset_tensor = get_reset_action(env)
                        obs, rew, terminated, truncated, info_ = env.step(reset_tensor)
            if (episode_idx >= total_episodes - 1 or actions is None) and reset_to_recorded_data:
                print(f"Reached the end of available episodes ({episode_idx + 1}/{total_episodes})")
                break

    infer_reader.stop()
    # close the simulator
    env.close()


if __name__ == "__main__":
    # run the main function
    main()
    # close sim app
    simulation_app.close()

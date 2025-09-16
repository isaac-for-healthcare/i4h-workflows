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

import gymnasium as gym
import numpy as np
import so_arm_starter_ext  # noqa: F401
import torch
from dds.publisher import Publisher
from dds.schemas.camera_info import CameraInfo
from dds.schemas.soarm_ctrl import SOARM101CtrlInput
from dds.schemas.soarm_info import SOARM101Info
from dds.subscriber import SubscriberWithQueue
from isaaclab.app import AppLauncher
from simulation.environments.state_machine.utils import capture_camera_images, get_joint_states

# add argparse arguments
parser = argparse.ArgumentParser(description="Run simulation in a single-arm manipulator, communication via DDS.")
parser.add_argument(
    "--disable_fabric", action="store_true", default=False, help="Disable fabric and use USD I/O operations."
)
parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to spawn.")
parser.add_argument(
    "--task",
    type=str,
    default="Isaac-SOARM101-v0",
    help="Name of the task.",
)
parser.add_argument(
    "--rti_license_file",
    type=str,
    default=os.getenv("RTI_LICENSE_FILE"),
    help="the path of rti_license_file. Default will use environment variables `RTI_LICENSE_FILE`",
)
parser.add_argument("--infer_domain_id", type=int, default=0, help="domain id to publish data for inference.")
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
    "--topic_in_soarm_pos",
    type=str,
    default="topic_soarm_info",
    help="topic name to consume soarm pos",
)
parser.add_argument(
    "--topic_out",
    type=str,
    default="topic_soarm_ctrl",
    help="topic name to publish generated soarm actions",
)
parser.add_argument(
    "--scale", type=float, default=1000.0, help="Scale factor to convert from omniverse to organ coordinate system."
)
parser.add_argument(
    "--hdf5_path",
    type=str,
    default=None,
    help="Path to single .hdf5 file or directory containing recorded data for environment reset.",
)

# append AppLauncher cli argruments
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()

if args_cli.rti_license_file is not None:
    if not os.path.isabs(args_cli.rti_license_file):
        raise ValueError("RTI license file must be an existing absolute path.")
    os.environ["RTI_LICENSE_FILE"] = args_cli.rti_license_file

app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app
reset_flag = False

from isaaclab_tasks.utils.parse_cfg import parse_env_cfg

pub_data = {
    "room_cam": None,
    "wrist_cam": None,
    "joint_pos": None,
}

hz = 60

# SO-ARM101 joint position limits in degrees for simulation
# These values are derived from the safety constraints
ISAACLAB_JOINT_POS_LIMIT_RANGE = [
    (-110.0, 110.0),
    (-100.0, 100.0),
    (-100.0, 90.0),
    (-95.0, 95.0),
    (-160.0, 160.0),
    (-10, 100.0),
]

# LEROBOT_JOINT_POS_LIMIT_RANGE: Normalized limits to match MotorNormMode.RANGE_M100_100 in Lerobot SO-ARM101
LEROBOT_JOINT_POS_LIMIT_RANGE = [
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (-100, 100),
    (0, 100),
]


def preprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Convert simulation joint positions to lerobot coordinate system."""
    joint_pos = joint_pos / np.pi * 180
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        joint_pos[:, i] = (joint_pos[:, i] - isaaclab_min) / (isaaclab_max - isaaclab_min) * (
            lerobot_max - lerobot_min
        ) + lerobot_min
    return joint_pos


def postprocess_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
    """Convert lerobot joint positions back to simulation coordinate system."""
    for i in range(6):
        isaaclab_min, isaaclab_max = ISAACLAB_JOINT_POS_LIMIT_RANGE[i]
        lerobot_min, lerobot_max = LEROBOT_JOINT_POS_LIMIT_RANGE[i]
        joint_pos[:, i] = (joint_pos[:, i] - lerobot_min) / (lerobot_max - lerobot_min) * (
            isaaclab_max - isaaclab_min
        ) + isaaclab_min
    joint_pos = joint_pos / 180 * np.pi
    return joint_pos


class RoomCamPublisher(Publisher):
    def __init__(self, topic: str, domain_id: int, rgb: bool = True):
        super().__init__(topic, CameraInfo, 1 / hz, domain_id)
        self.rgb = rgb

    def produce(self, dt: float, sim_time: float):
        output = CameraInfo()
        output.focal_len = 12.0
        output.height = 480
        output.width = 640
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
        output.height = 480
        output.width = 640
        if self.rgb:
            output.data = pub_data["wrist_cam"].tobytes()
        else:
            output.data = pub_data["wrist_cam_depth"].tobytes()
        return output


class PosPublisher(Publisher):
    def __init__(self, domain_id: int):
        super().__init__(args_cli.topic_in_soarm_pos, SOARM101Info, 1 / hz, domain_id)

    def produce(self, dt: float, sim_time: float):
        output = SOARM101Info()
        output.joints_state_positions = pub_data["joint_pos"].tolist()
        return output


def get_reset_action(env, use_rel: bool = False):
    """Get the reset action using custom joint positions."""
    reset_joint_positions = torch.tensor([0.0, -1.6, 1.4, 1.5, -1.8, 0.0], device=args_cli.device)

    # Repeat for all environments and add batch dimension
    reset_tensor = reset_joint_positions.unsqueeze(0).repeat(env.unwrapped.num_envs, 1)

    return reset_tensor


@torch.inference_mode()
def main():
    """Main function."""

    env_cfg = parse_env_cfg(
        args_cli.task, device=args_cli.device, num_envs=args_cli.num_envs, use_fabric=not args_cli.disable_fabric
    )

    # modify configuration
    env_cfg.terminations.time_out = None
    env_cfg.use_teleop_device("so101leader")

    env = gym.make(args_cli.task, cfg=env_cfg).unwrapped

    print(f"[INFO]: Gym observation space: {env.observation_space}")
    print(f"[INFO]: Gym action space: {env.action_space}")

    env.reset()
    for _ in range(50):
        reset_tensor = get_reset_action(env)
        obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    max_timesteps = 500
    action_dim = 6

    infer_r_cam_writer = RoomCamPublisher(topic=args_cli.topic_in_room_camera, domain_id=args_cli.infer_domain_id)
    infer_w_cam_writer = WristCamPublisher(topic=args_cli.topic_in_wrist_camera, domain_id=args_cli.infer_domain_id)
    infer_pos_writer = PosPublisher(args_cli.infer_domain_id)
    # SOARM101CtrlInput is action predicted by gr00t
    infer_reader = SubscriberWithQueue(args_cli.infer_domain_id, args_cli.topic_out, SOARM101CtrlInput, 1 / hz)
    infer_reader.start()

    total_episodes = 1

    while simulation_app.is_running():
        global reset_flag
        for episode_idx in range(total_episodes):
            print(f"\nepisode_idx: {episode_idx}")

            action_plan = collections.deque()

            for t in range(max_timesteps):
                # get and publish the current images and joint positions
                rgb_images, _ = capture_camera_images(env, ["room", "wrist"], device=env.unwrapped.device)

                (pub_data["room_cam"],) = (rgb_images[0, 0, ...].cpu().numpy(),)
                (pub_data["wrist_cam"],) = (rgb_images[0, 1, ...].cpu().numpy(),)
                joint_pos = get_joint_states(env)[0]

                if joint_pos.ndim == 1:
                    joint_pos = joint_pos.reshape(1, -1)
                processed_joint_pos = preprocess_joint_pos(joint_pos)  # rads to degrees
                pub_data["joint_pos"] = processed_joint_pos.flatten()
                # should equal to current joint pos in policy runner

                if not action_plan:
                    # publish the images and joint positions when run policy inference
                    infer_r_cam_writer.write()
                    infer_w_cam_writer.write()
                    infer_pos_writer.write()

                    ret = None
                    while ret is None:
                        ret = infer_reader.read_data()
                    o: SOARM101CtrlInput = ret

                    action_chunk = np.array(o.joint_positions, dtype=np.float32).reshape(-1, action_dim)
                    action_plan.extend(action_chunk)

                action = action_plan.popleft()
                # Ensure action is 2D for postprocess_joint_pos function
                if action.ndim == 1:
                    action = action.reshape(1, -1)
                action = postprocess_joint_pos(action)
                # Flatten back to 1D for torch tensor conversion
                action = action.flatten().astype(np.float32)
                action = torch.tensor(action, device=env.unwrapped.device).repeat(env.unwrapped.num_envs, 1)

                obs, rew, terminated, truncated, info_ = env.step(action)

            env.reset()
            for _ in range(50):
                reset_tensor = get_reset_action(env)
                obs, rew, terminated, truncated, info_ = env.step(reset_tensor)

    infer_reader.stop()
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()

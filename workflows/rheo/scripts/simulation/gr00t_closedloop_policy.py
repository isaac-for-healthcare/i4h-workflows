# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from pathlib import Path
from typing import Any

import gymnasium as gym
import numpy as np
import torch
import yaml
from gr00t.data.embodiment_tags import EmbodimentTag
from gr00t.policy.gr00t_policy import Gr00tPolicy
from isaaclab_arena.policy.policy_base import PolicyBase
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.data_utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.data_utils.io_utils import create_config_from_yaml, load_robot_joints_config_from_yaml
from isaaclab_arena_gr00t.data_utils.joints_conversion import remap_sim_joints_to_policy_joints
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition
from isaaclab_arena_gr00t.policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from scripts.utils.joint_conversion import remap_policy_joints_to_sim_joints
from scripts.utils.policy_tasks import replace_dit_with_tensorrt


class CustomGr00tClosedloopPolicy(PolicyBase):
    def __init__(self, policy_config_yaml_path: Path, num_envs: int = 1, device: str = "cuda"):
        """
        Base class for closedloop inference from obs using GR00T N1.5 policy

        Args:
            policy_config_yaml_path: Path to policy config YAML
            num_envs: Number of parallel environments
            device: Device to run on (cuda/cpu)
        """
        self.policy_config = create_config_from_yaml(policy_config_yaml_path, Gr00tClosedloopPolicyConfig)
        with open(policy_config_yaml_path, "r") as f:
            yaml_data = yaml.safe_load(f)
        self.trt_engine_path = yaml_data.get("trt_engine_path", None)

        self.policy = self.load_policy()

        # determine rollout how many action prediction per observation
        self.action_chunk_length = self.policy_config.action_chunk_length

        self.num_envs = num_envs
        self.device = device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        self.policy_joints_config = self.load_policy_joints_config(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = self.load_sim_action_joints_config(
            self.policy_config.action_joints_config_path
        )
        self.robot_state_joints_config = self.load_sim_state_joints_config(self.policy_config.state_joints_config_path)

        # action_dim = joints + WBC commands (for G1 locomanipulation)
        self.action_dim = len(self.robot_action_joints_config)
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            self.action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD

        # action chunking states (reference implementation)
        self.current_action_chunk = torch.zeros(
            (num_envs, self.policy_config.action_horizon, self.action_dim),
            dtype=torch.float,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def load_policy_joints_config(self, policy_config_path: Path) -> dict[str, Any]:
        """Load the GR00T policy joint config from the data config."""
        return load_robot_joints_config_from_yaml(policy_config_path)

    def load_sim_state_joints_config(self, state_config_path: Path) -> dict[str, Any]:
        """Load the simulation state joint config from the data config."""
        return load_robot_joints_config_from_yaml(state_config_path)

    def load_sim_action_joints_config(self, action_config_path: Path) -> dict[str, Any]:
        """Load the simulation action joint config from the data config."""
        return load_robot_joints_config_from_yaml(action_config_path)

    def load_policy(self) -> Gr00tPolicy:
        """Load the policy and optionally replace DiT with TensorRT."""
        assert Path(
            self.policy_config.model_path
        ).exists(), f"Model path {self.policy_config.model_path} does not exist"

        # Convert embodiment_tag string to EmbodimentTag enum
        embodiment_tag_enum = EmbodimentTag(self.policy_config.embodiment_tag)

        policy = Gr00tPolicy(
            embodiment_tag=embodiment_tag_enum,  # ← Use enum, not string!
            model_path=self.policy_config.model_path,
            device=self.policy_config.policy_device,
            strict=True,
        )

        # Apply TensorRT or PyTorch mode
        if self.trt_engine_path is not None:
            if not Path(self.trt_engine_path).exists():
                raise FileNotFoundError(f"TensorRT engine path {self.trt_engine_path} does not exist")

            device_idx = 0
            if ":" in self.policy_config.policy_device:
                device_idx = int(self.policy_config.policy_device.split(":")[-1])
            replace_dit_with_tensorrt(policy, self.trt_engine_path, device=device_idx)
            print("[TRT] Using TensorRT inference mode")
        else:
            # PyTorch mode with torch.compile (optional optimization)
            print("[PyTorch] Using PyTorch inference mode")

        return policy

    def get_observations(self, observation: dict[str, Any], camera_name: str = "robot_head_cam_rgb") -> dict[str, Any]:
        """Build GR00T policy observation dict (single-camera).

        We only support a single RGB camera in this project setup. This avoids any
        camera/video-key matching logic.
        """

        if camera_name not in observation["camera_obs"]:
            raise ValueError(f"Camera {camera_name} not found in observation: {observation['camera_obs'].keys()}")

        # Pick the single LeRobot video key to populate. If the config provides a list,
        # we take the first entry; otherwise fall back to a sane default.
        video_key_full = "video.ego_view"
        if hasattr(self.policy_config, "video_keys_lerobot"):
            keys = getattr(self.policy_config, "video_keys_lerobot")
            if isinstance(keys, (list, tuple)) and len(keys) > 0 and isinstance(keys[0], str):
                video_key_full = keys[0]

        rgb = observation["camera_obs"][camera_name]
        # gr00t uses numpy arrays
        rgb = rgb.cpu().numpy()
        # Apply preprocessing to rgb if size is not the same as the target size
        if rgb.shape[1:3] != self.policy_config.target_image_size[:2]:
            rgb = resize_frames_with_padding(
                rgb, target_image_size=self.policy_config.target_image_size, bgr_conversion=False, pad_img=True
            )

        camera_data = {
            video_key_full: rgb.reshape(
                self.num_envs,
                1,
                self.policy_config.target_image_size[0],
                self.policy_config.target_image_size[1],
                self.policy_config.target_image_size[2],
            )
        }

        # GR00T uses np arrays, needs to copy torch tensor from gpu to cpu before conversion
        joint_pos_sim = observation["policy"]["robot_joint_pos"].cpu()
        joint_pos_state_sim = JointsAbsPosition(joint_pos_sim, self.robot_state_joints_config)
        # Retrieve joint positions as proprioceptive states and remap to policy joint orders
        joint_pos_state_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, self.policy_joints_config)

        # Pack inputs to dictionary in nested format (required by Gr00tPolicy)
        state_data = {
            "left_arm": joint_pos_state_policy["left_arm"].reshape(self.num_envs, 1, -1),
            "right_arm": joint_pos_state_policy["right_arm"].reshape(self.num_envs, 1, -1),
            "left_hand": joint_pos_state_policy["left_hand"].reshape(self.num_envs, 1, -1),
            "right_hand": joint_pos_state_policy["right_hand"].reshape(self.num_envs, 1, -1),
        }
        # NOTE(xinjieyao, 2025-10-07): waist is not used in GR1 tabletop manipulation
        # For G1 locomanipulation, add all body parts including legs and waist
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            state_data["left_leg"] = joint_pos_state_policy["left_leg"].reshape(self.num_envs, 1, -1)
            state_data["right_leg"] = joint_pos_state_policy["right_leg"].reshape(self.num_envs, 1, -1)
            state_data["waist"] = joint_pos_state_policy["waist"].reshape(self.num_envs, 1, -1)

        # Extract video keys from camera_data (remove "video." prefix)
        video_data = {}
        for key, value in camera_data.items():
            # key is like "video.ego_view", extract "ego_view"
            if key.startswith("video."):
                video_key = key.replace("video.", "")
                video_data[video_key] = value
            else:
                video_data[key] = value

        # Create nested structure required by Gr00tPolicy
        policy_observations = {
            "video": video_data,
            "state": state_data,
            "language": {
                "annotation.human.task_description": [[self.policy_config.language_instruction]] * self.num_envs
            },
        }
        return policy_observations

    def get_action(self, env: gym.Env, observation: dict[str, Any]) -> torch.Tensor:
        """Get the immediate next action from the current action chunk.

        If the action chunk is not yet computed, compute a new action chunk first before returning the action.

        Returns:
            action: The immediate next action to execute per env.step() call. Shape: (num_envs, action_dim)
        """
        # get action chunk if not yet computed
        if any(self.env_requires_new_action_chunk):
            # compute a new action chunk for the envs that require a new action chunk
            returned_action_chunk = self.get_action_chunk(observation, self.policy_config.pov_cam_name_sim)
            self.current_action_chunk[self.env_requires_new_action_chunk] = returned_action_chunk[
                self.env_requires_new_action_chunk
            ]
            # reset the action index for those env_ids
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            # reset the env_requires_new_action_chunk for those env_ids
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        # assert for all env_ids that the action index is valid
        assert self.current_action_index.min() >= 0, "At least one env's action index is less than 0"
        assert (
            self.current_action_index.max() < self.action_chunk_length
        ), "At least one env's action index is greater than the action chunk length"

        # for i-th row in action_chunk, use the value of i-th element in current_action_index
        # to select the action from the action chunk
        action = self.current_action_chunk[torch.arange(self.num_envs), self.current_action_index]
        assert action.shape == (
            self.num_envs,
            self.action_dim,
        ), f"{action.shape=} != ({self.num_envs}, {self.action_dim})"

        self.current_action_index += 1

        # for those rows in current_action_chunk that equal to action_chunk_length, reset to 0
        reset_env_ids = self.current_action_index == self.action_chunk_length
        self.current_action_chunk[reset_env_ids] = 0.0
        # indicate that the action chunk is not yet computed for those env_ids
        self.env_requires_new_action_chunk[reset_env_ids] = True
        # set the action index for those env_ids to -1 to indicate that the action chunk is reset
        self.current_action_index[reset_env_ids] = -1

        return action

    def get_action_chunk(self, observation: dict[str, Any], camera_name: str = "robot_head_cam_rgb") -> torch.Tensor:
        policy_observations = self.get_observations(observation, camera_name)
        # print(f"policy_observations: {policy_observations.keys()}\n")
        # get_action returns (action_dict, info_dict)
        robot_action_policy, info = self.policy.get_action(policy_observations)
        # print(f"robot_action_policy: {robot_action_policy.keys()}\n")
        # print(f"robot_action_policy shapes: {[(k, v.shape) for k, v in robot_action_policy.items()]}\n")
        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy, self.policy_joints_config, self.robot_action_joints_config, self.device
        )
        # print(f"robot_action_sim.get_joints_pos(): {robot_action_sim.get_joints_pos().shape}\n")

        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            # NOTE(xinjieyao, 2025-09-29): GR00T output dim=32, does not fit the entire action space,
            # including torso_orientation_rpy_command. Manually set it to 0.
            torso_orientation_rpy_command = np.zeros(robot_action_policy["navigate_command"].shape)
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.from_numpy(robot_action_policy["navigate_command"]).to(self.device),
                    torch.from_numpy(robot_action_policy["base_height_command"]).to(self.device),
                    torch.from_numpy(torso_orientation_rpy_command).to(self.device),
                ],
                axis=2,
            )
        elif self.task_mode == TaskMode.GR1_TABLETOP_MANIPULATION:
            action_tensor = robot_action_sim.get_joints_pos()

        assert action_tensor.shape[0] == self.num_envs and action_tensor.shape[1] >= self.action_chunk_length
        return action_tensor.float()  # ensure float32 to match current_action_chunk dtype

    def reset(self, env_ids: torch.Tensor | None = None):
        """
        Resets the action chunking mechanism. As GR00T policy predicts a sequence of future
        low-level actions in a single forward pass, we don't need to reset its internal state.
        It zeros the action chunk, sets the action index to -1, and sets the
        boolean indicator env_requires_new_action_chunk to True for the required env_ids.

        Args:
            env_ids: the env_ids to reset. If None, reset all envs.
        """
        if env_ids is None:
            env_ids = slice(None)
        self.current_action_chunk[env_ids] = 0.0
        self.current_action_index[env_ids] = -1
        self.env_requires_new_action_chunk[env_ids] = True

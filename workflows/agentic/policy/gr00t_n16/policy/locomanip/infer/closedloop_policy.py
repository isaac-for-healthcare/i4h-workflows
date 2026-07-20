# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from dataclasses import MISSING, fields
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import torch
import yaml
from isaaclab_arena_g1.g1_whole_body_controller.wbc_policy.policy.policy_constants import (
    NUM_BASE_HEIGHT_CMD,
    NUM_NAVIGATE_CMD,
    NUM_TORSO_ORIENTATION_RPY_CMD,
)
from isaaclab_arena_gr00t.data_utils.image_conversion import resize_frames_with_padding
from isaaclab_arena_gr00t.data_utils.io_utils import load_robot_joints_config_from_yaml
from isaaclab_arena_gr00t.data_utils.joints_conversion import remap_sim_joints_to_policy_joints
from isaaclab_arena_gr00t.data_utils.robot_joints import JointsAbsPosition
from isaaclab_arena_gr00t.policy_config import Gr00tClosedloopPolicyConfig, TaskMode
from policy.locomanip.infer.joint_conversion import remap_policy_joints_to_sim_joints
from policy.locomanip.infer.tensorrt_dit import replace_dit_with_tensorrt

logger = logging.getLogger("policy")
_AGENTIC_POLICY_CONFIG_FIELDS = {"pov_cam_names_sim"}


class G1LocomanipClosedloopPolicy:
    def __init__(
        self,
        policy_config_yaml_path: Path | None = None,
        num_envs: int = 1,
        device: str = "cuda",
        model_path_override: str | None = None,
        model_repo: str | None = None,
        model_revision: str | None = None,
        policy_config_data: dict[str, Any] | None = None,
        config_base_dir: Path | None = None,
    ):
        self.config_path = Path(policy_config_yaml_path).resolve() if policy_config_yaml_path else None
        self.config_base_dir = (
            self.config_path.parent
            if self.config_path is not None
            else Path(config_base_dir).resolve()
            if config_base_dir is not None
            else Path.cwd()
        )
        self.model_path_override = model_path_override
        self.model_repo = model_repo
        self.model_revision = model_revision
        yaml_data = self._load_yaml_config() if self.config_path is not None else dict(policy_config_data or {})
        yaml_data = self._resolve_config_paths(yaml_data)
        self.trt_engine_path = yaml_data.get("trt_engine_path", None)
        self.policy_config = self._build_policy_config(yaml_data)
        self.resolved_model_path: Path | None = None
        self.policy = self.load_policy()
        self.action_chunk_length = self.policy_config.action_chunk_length
        self.num_envs = num_envs
        self.device = device
        self.task_mode = TaskMode(self.policy_config.task_mode_name)

        self.policy_joints_config = self.load_policy_joints_config(self.policy_config.policy_joints_config_path)
        self.robot_action_joints_config = self.load_sim_action_joints_config(
            self.policy_config.action_joints_config_path
        )
        self.robot_state_joints_config = self.load_sim_state_joints_config(self.policy_config.state_joints_config_path)

        self.action_dim = len(self.robot_action_joints_config)
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            self.action_dim += NUM_NAVIGATE_CMD + NUM_BASE_HEIGHT_CMD + NUM_TORSO_ORIENTATION_RPY_CMD

        self.current_action_chunk = torch.zeros(
            (num_envs, self.policy_config.action_horizon, self.action_dim),
            dtype=torch.float,
            device=device,
        )
        self.env_requires_new_action_chunk = torch.ones(num_envs, dtype=torch.bool, device=device)
        self.current_action_index = torch.zeros(num_envs, dtype=torch.int64, device=device)

    def _load_yaml_config(self) -> dict[str, Any]:
        if self.config_path is None:
            return {}
        with open(self.config_path) as f:
            return yaml.safe_load(f) or {}

    def _resolve_config_paths(self, yaml_data: dict[str, Any]) -> dict[str, Any]:
        for key in (
            "model_path",
            "policy_joints_config_path",
            "action_joints_config_path",
            "state_joints_config_path",
            "trt_engine_path",
        ):
            if yaml_data.get(key):
                yaml_data[key] = str(self._resolve_config_path(yaml_data[key]))
        return yaml_data

    def _config_fields(self, yaml_data: dict[str, Any]) -> dict[str, Any]:
        config_fields = set(Gr00tClosedloopPolicyConfig.__dataclass_fields__)
        config_fields.update(_AGENTIC_POLICY_CONFIG_FIELDS)
        return {key: value for key, value in yaml_data.items() if key in config_fields}

    def _build_policy_config(self, yaml_data: dict[str, Any]) -> SimpleNamespace:
        config_data = {}
        for field in fields(Gr00tClosedloopPolicyConfig):
            if field.default is not MISSING:
                config_data[field.name] = field.default
            elif field.default_factory is not MISSING:
                config_data[field.name] = field.default_factory()
        config_data.update(self._config_fields(yaml_data))
        return SimpleNamespace(**config_data)

    def _resolve_config_path(self, path: Path | str) -> Path:
        candidate = Path(path)
        if candidate.is_absolute():
            return candidate
        return (self.config_base_dir / candidate).resolve()

    def load_policy_joints_config(self, environment_config_path: Path) -> dict[str, Any]:
        return load_robot_joints_config_from_yaml(self._resolve_config_path(environment_config_path))

    def load_sim_state_joints_config(self, state_config_path: Path) -> dict[str, Any]:
        return load_robot_joints_config_from_yaml(self._resolve_config_path(state_config_path))

    def load_sim_action_joints_config(self, action_config_path: Path) -> dict[str, Any]:
        return load_robot_joints_config_from_yaml(self._resolve_config_path(action_config_path))

    def load_policy(self) -> Any:
        model_path = self._resolve_model_path()
        self.resolved_model_path = model_path
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag(self.policy_config.embodiment_tag),
            model_path=str(model_path),
            device=self.policy_config.policy_device,
            strict=True,
        )
        if self.trt_engine_path is not None:
            trt_path = self._resolve_config_path(self.trt_engine_path)
            if not trt_path.exists():
                raise FileNotFoundError(f"TensorRT engine path {trt_path} does not exist")
            device_idx = 0
            if ":" in self.policy_config.policy_device:
                device_idx = int(self.policy_config.policy_device.split(":")[-1])
            replace_dit_with_tensorrt(policy, str(trt_path), device=device_idx)
        return policy

    def _resolve_model_path(self) -> Path:
        if self.model_path_override:
            model_path = Path(self.model_path_override).expanduser()
            if not model_path.exists():
                raise FileNotFoundError(f"G1 --model-path not found: {model_path}")
            logger.info("Using G1 model from --model-path: %s", model_path)
            return model_path

        config_model_path = getattr(self.policy_config, "model_path", None)
        if config_model_path:
            model_path = Path(config_model_path).expanduser()
            if model_path.exists():
                logger.info("Using G1 model from config: %s", model_path)
                return model_path

        if not self.model_repo:
            raise FileNotFoundError(
                "No G1 model repo configured. Pass --model-path, --g1-model-repo, or set model_repo in config/environments/<env>.yaml."
            )

        from huggingface_hub import snapshot_download

        cache_dir = os.environ.get("RHEO_MODEL_CACHE", os.path.expanduser("~/.cache/rheo_models"))
        os.makedirs(cache_dir, exist_ok=True)
        logger.info(
            "Pulling G1 model repo %s revision=%s into %s",
            self.model_repo,
            self.model_revision or "<default>",
            cache_dir,
        )
        return Path(snapshot_download(repo_id=self.model_repo, revision=self.model_revision, cache_dir=cache_dir))

    def get_observations(self, observation: dict[str, Any]) -> dict[str, Any]:
        joint_pos_sim = observation["policy"]["robot_joint_pos"].cpu()
        joint_pos_state_sim = JointsAbsPosition(joint_pos_sim, self.robot_state_joints_config)
        joint_pos_state_policy = remap_sim_joints_to_policy_joints(joint_pos_state_sim, self.policy_joints_config)

        state_data = {
            "left_arm": joint_pos_state_policy["left_arm"].reshape(self.num_envs, 1, -1),
            "right_arm": joint_pos_state_policy["right_arm"].reshape(self.num_envs, 1, -1),
            "left_hand": joint_pos_state_policy["left_hand"].reshape(self.num_envs, 1, -1),
            "right_hand": joint_pos_state_policy["right_hand"].reshape(self.num_envs, 1, -1),
        }
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            state_data["left_leg"] = joint_pos_state_policy["left_leg"].reshape(self.num_envs, 1, -1)
            state_data["right_leg"] = joint_pos_state_policy["right_leg"].reshape(self.num_envs, 1, -1)
            state_data["waist"] = joint_pos_state_policy["waist"].reshape(self.num_envs, 1, -1)

        return {
            "video": self._video_data(observation["camera_obs"]),
            "state": state_data,
            "language": {
                "annotation.human.task_description": [[self.policy_config.language_instruction]] * self.num_envs
            },
        }

    def _video_data(self, camera_obs: dict[str, torch.Tensor]) -> dict[str, np.ndarray]:
        video_data = {}
        for obs_key, video_key in self._camera_specs():
            if obs_key not in camera_obs:
                raise ValueError(f"Camera {obs_key} not found in observation: {list(camera_obs.keys())}")
            video_data[video_key] = self._format_video_frame(camera_obs[obs_key])
        return video_data

    def _camera_specs(self) -> list[tuple[str, str]]:
        cam_specs = getattr(self.policy_config, "pov_cam_names_sim", None)
        if not cam_specs:
            raise ValueError("policy.pov_cam_names_sim must define at least one camera mapping")
        return [
            (spec["obs_key"], spec["video_key"]) if isinstance(spec, dict) else (spec[0], spec[1]) for spec in cam_specs
        ]

    def _format_video_frame(self, frame: torch.Tensor) -> np.ndarray:
        rgb = frame.cpu().numpy()
        target_image_size = self.policy_config.target_image_size
        if rgb.shape[1:3] != target_image_size[:2]:
            rgb = resize_frames_with_padding(
                rgb,
                target_image_size=target_image_size,
                bgr_conversion=False,
                pad_img=True,
            )
        return rgb.reshape(self.num_envs, 1, target_image_size[0], target_image_size[1], target_image_size[2])

    def get_action(self, observation: dict[str, Any]) -> torch.Tensor:
        if any(self.env_requires_new_action_chunk):
            returned_action_chunk = self.get_action_chunk(observation)
            self.current_action_chunk[self.env_requires_new_action_chunk] = returned_action_chunk[
                self.env_requires_new_action_chunk
            ]
            self.current_action_index[self.env_requires_new_action_chunk] = 0
            self.env_requires_new_action_chunk[self.env_requires_new_action_chunk] = False

        action = self.current_action_chunk[torch.arange(self.num_envs), self.current_action_index]
        self.current_action_index += 1
        reset_env_ids = self.current_action_index == self.action_chunk_length
        self.current_action_chunk[reset_env_ids] = 0.0
        self.env_requires_new_action_chunk[reset_env_ids] = True
        self.current_action_index[reset_env_ids] = -1
        return action

    def get_action_chunk(self, observation: dict[str, Any]) -> torch.Tensor:
        policy_observations = self.get_observations(observation)
        robot_action_policy, _info = self.policy.get_action(policy_observations)
        robot_action_sim = remap_policy_joints_to_sim_joints(
            robot_action_policy,
            self.policy_joints_config,
            self.robot_action_joints_config,
            self.device,
        )
        if self.task_mode == TaskMode.G1_LOCOMANIPULATION:
            navigate_command = self._action_value(robot_action_policy, "navigate_command", "action.navigate_command")
            base_height_command = self._action_value(
                robot_action_policy, "base_height_command", "action.base_height_command"
            )
            torso_orientation_rpy_command = np.zeros(navigate_command.shape)
            action_tensor = torch.cat(
                [
                    robot_action_sim.get_joints_pos(),
                    torch.as_tensor(navigate_command, dtype=torch.float, device=self.device),
                    torch.as_tensor(base_height_command, dtype=torch.float, device=self.device),
                    torch.from_numpy(torso_orientation_rpy_command).to(self.device),
                ],
                axis=2,
            )
        else:
            action_tensor = robot_action_sim.get_joints_pos()
        return action_tensor.float()

    @staticmethod
    def _action_value(action: dict[str, Any], *keys: str) -> np.ndarray:
        for key in keys:
            if key in action:
                return action[key]
        raise KeyError(f"missing action key; tried {keys}, available={sorted(action.keys())}")

    def reset(self) -> None:
        self.current_action_chunk[:] = 0.0
        self.current_action_index[:] = -1
        self.env_requires_new_action_chunk[:] = True


CustomGr00tClosedloopPolicy = G1LocomanipClosedloopPolicy

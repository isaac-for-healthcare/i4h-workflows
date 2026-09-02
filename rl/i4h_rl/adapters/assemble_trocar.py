# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T/RLinf mapping for the maintained Workflow Trocar Scene."""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_registered = False

OBS_CONVERTER = "i4h_g1_dex3"
TRAIN_TASK_ID = "I4H-Workflows-Assemble-Trocar-RLinf-v0"
EVAL_TASK_ID = "I4H-Workflows-Assemble-Trocar-RLinf-Eval-v0"
ACTION_KEYS = ("action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand")


def _to_rgb(image: Any) -> Any:
    return image[..., :3]


def wrap_workflow_observation(obs: dict[str, Any], *, task_description: str, num_envs: int) -> dict[str, Any]:
    """Map current Workflow observation terms to the small RLinf bridge schema."""
    import torch

    policy = obs.get("policy", obs)
    required = (
        "front_camera_rgb",
        "left_wrist_camera_rgb",
        "right_wrist_camera_rgb",
        "robot_joint_state",
        "robot_dex3_joint_state",
    )
    missing = [name for name in required if name not in policy]
    if missing:
        raise KeyError(f"Workflow Trocar observation is missing {missing}; got {sorted(policy)}")
    body = policy["robot_joint_state"]
    hands = policy["robot_dex3_joint_state"]
    if body.shape[-1] != 87 or hands.shape[-1] != 14:
        raise ValueError(f"expected G1 body/hand state widths 87/14, got {body.shape[-1]}/{hands.shape[-1]}")
    return {
        "main_images": _to_rgb(policy["front_camera_rgb"]),
        "extra_view_images": torch.stack(
            (_to_rgb(policy["left_wrist_camera_rgb"]), _to_rgb(policy["right_wrist_camera_rgb"])), dim=1
        ),
        "states": torch.cat((body[:, 15:29], hands), dim=-1),
        "task_descriptions": [task_description] * num_envs,
    }


def convert_workflow_obs_to_gr00t(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert the bridge schema to the GR00T N1.5 Trocar modality contract."""
    import torch

    main = env_obs["main_images"]
    extra = env_obs["extra_view_images"]
    states = env_obs["states"]
    if not all(isinstance(value, torch.Tensor) for value in (main, extra, states)):
        raise TypeError("Workflow Trocar images and states must be torch tensors")
    state = states.unsqueeze(1).cpu().numpy()
    return {
        "video.room_view": main.unsqueeze(1).cpu().numpy(),
        "video.left_wrist_view": extra[:, 0].unsqueeze(1).cpu().numpy(),
        "video.right_wrist_view": extra[:, 1].unsqueeze(1).cpu().numpy(),
        "state.left_arm": state[:, :, 0:7],
        "state.right_arm": state[:, :, 7:14],
        "state.left_hand": state[:, :, 14:21],
        "state.right_hand": state[:, :, 21:28],
        "annotation.human.task_description": env_obs["task_descriptions"],
    }


def convert_gr00t_to_workflow_action(action_chunk: dict[str, Any], chunk_size: int = 1) -> np.ndarray:
    """Map GR00T's 28 arm/hand joints into Workflow's 43-DoF G1 action."""
    missing = [key for key in ACTION_KEYS if key not in action_chunk]
    if missing:
        raise KeyError(f"GR00T Trocar action is missing {missing}; got {sorted(action_chunk)}")
    controlled = np.concatenate([np.asarray(action_chunk[key])[:, :chunk_size, :] for key in ACTION_KEYS], axis=-1)
    if controlled.shape[-1] != 28:
        raise ValueError(f"expected 28 controlled arm/hand joints, got {controlled.shape[-1]}")
    return np.pad(controlled, ((0, 0), (0, 0), (15, 0)), mode="constant")


def _register_gr00t_converters(simulation_io: Any) -> None:
    """Register this N1.5 environment against the pinned RLinf registries."""
    try:
        action_registry = simulation_io.ACTION_CONVERSION_N1D5
    except AttributeError as exc:
        raise RuntimeError("pinned RLinf does not expose the GR00T N1.5 action registry") from exc
    simulation_io.OBS_CONVERSION[OBS_CONVERTER] = convert_workflow_obs_to_gr00t
    action_registry[OBS_CONVERTER] = convert_gr00t_to_workflow_action


def _install_gr00t_n15_data_config_loader() -> None:
    """Provide the import-string loader expected by the IsaacLab adapter."""
    from gr00t.experiment import data_config as gr00t_data_config

    if hasattr(gr00t_data_config, "load_data_config"):
        return

    def load_data_config(specification: str):
        module_name, separator, attribute_name = specification.partition(":")
        if separator:
            config_type = getattr(importlib.import_module(module_name), attribute_name)
            return config_type()
        try:
            config = gr00t_data_config.DATA_CONFIG_MAP[specification]
        except KeyError as exc:
            raise ValueError(f"unknown GR00T data config: {specification}") from exc
        return config() if isinstance(config, type) else config

    gr00t_data_config.load_data_config = load_data_config


def _install_rlinf_n15_module_alias() -> None:
    """Bridge the pre-versioned IsaacLab import to RLinf's N1.5 package."""
    legacy_name = "rlinf.models.embodiment.gr00t.gr00t_action_model"
    if legacy_name in sys.modules:
        return
    current_name = "rlinf.models.embodiment.gr00t.gr00t_n1d5.gr00t_action_model"
    sys.modules[legacy_name] = importlib.import_module(current_name)


def _get_workflow_env_class():
    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class WorkflowTrocarEnv(IsaaclabBaseEnv):
        def _init_isaaclab_env(self):
            from i4h_rl.sim_bridge import RemoteIsaacEnv

            self.env = RemoteIsaacEnv.from_environment()
            self.env.reset(seed=self.seed)

        def _wrap_obs(self, obs):
            return wrap_workflow_observation(
                obs,
                task_description=self.task_description,
                num_envs=self.num_envs,
            )

        def _record_metrics(self, step_reward, terminations, infos):
            episode_info = {}
            self.returns += step_reward
            self.success_once = self.success_once | terminations.bool()
            episode_info["success_once"] = self.success_once.clone()
            episode_info["return"] = self.returns.clone()
            episode_info["episode_len"] = self.elapsed_steps.clone()
            episode_info["reward"] = episode_info["return"] / episode_info["episode_len"]
            infos["episode"] = episode_info
            return infos

        def add_image(self, obs):
            policy = obs.get("policy", obs)
            image = policy.get("front_camera_rgb")
            return None if image is None else _to_rgb(image[0]).cpu().numpy()

    return WorkflowTrocarEnv


def register() -> None:
    """Register Trocar conversion, model loading, and environment factories."""
    global _registered
    if _registered:
        return

    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    env_class = _get_workflow_env_class()
    REGISTER_ISAACLAB_ENVS[TRAIN_TASK_ID] = env_class
    REGISTER_ISAACLAB_ENVS[EVAL_TASK_ID] = env_class

    from isaaclab_contrib.rl.rlinf import extension as isaaclab_extension
    from rlinf.models.embodiment.gr00t import simulation_io

    cfg = isaaclab_extension._get_isaaclab_cfg()
    if cfg.get("obs_converter_type") != OBS_CONVERTER:
        raise ValueError(f"expected obs_converter_type={OBS_CONVERTER!r}, got {cfg.get('obs_converter_type')!r}")
    _register_gr00t_converters(simulation_io)
    _install_gr00t_n15_data_config_loader()
    _install_rlinf_n15_module_alias()
    isaaclab_extension._patch_gr00t_get_model(cfg)
    _registered = True
    logger.info("registered Workflow Trocar RL tasks: %s, %s", TRAIN_TASK_ID, EVAL_TASK_ID)

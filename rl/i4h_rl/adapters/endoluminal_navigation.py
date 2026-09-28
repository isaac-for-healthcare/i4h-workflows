# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T/RLinf mapping for the catheter navigation Scene.

This post-trains a checkpoint already fine-tuned on catheter demonstrations.
The state and action widths below are the catheter's, not a generic robot's,
and a checkpoint trained against any other modality will fail to load rather
than train badly -- which is the preferable failure.

Two things differ from the Trocar adapter and are worth stating.

There is one camera. A C-arm is the only view an interventionalist has, so
``extra_view_images`` is empty rather than padded with a second angle that
does not exist clinically. The bridge schema keeps the key so the shape of a
log line does not change between workflows.

Nothing is padded on the action side. The G1 adapter maps 28 controlled
joints into a 43-DoF action and left-pads the rest; every one of the
catheter's four channels is commanded, so the mapping is the identity and a
pad would silently zero a real control.
"""

from __future__ import annotations

import importlib
import logging
import sys
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)
_registered = False

OBS_CONVERTER = "i4h_catheter_carm"
TRAIN_TASK_ID = "I4H-Workflows-Endoluminal-Navigation-RLinf-v0"
EVAL_TASK_ID = "I4H-Workflows-Endoluminal-Navigation-RLinf-Eval-v0"

#: Ordered as the trainer config's ``state`` slices expect. Changing this
#: order silently re-labels GR00T's state groups, so it is asserted below
#: rather than trusted.
STATE_KEYS = ("drive_state", "tip_position", "tip_direction", "target_offset", "route_state")
STATE_WIDTHS = (4, 3, 3, 3, 2)
STATE_DIM = sum(STATE_WIDTHS)

#: Insertion velocity, axial rotation rate, tip bend rate, C-arm orbit rate.
ACTION_KEYS = ("action.catheter", "action.carm")
ACTION_WIDTHS = (3, 1)
ACTION_DIM = sum(ACTION_WIDTHS)


def _to_rgb(image: Any) -> Any:
    return image[..., :3]


def wrap_workflow_observation(obs: dict[str, Any], *, task_description: str, num_envs: int) -> dict[str, Any]:
    """Map current Workflow observation terms to the small RLinf bridge schema.

    The state is concatenated here, in the documented ``STATE_KEYS`` order,
    so that the trainer config's slices have one place to agree with.
    """
    import torch

    policy = obs.get("policy", obs)
    required = ("fluoroscopy_rgb", *STATE_KEYS)
    missing = [name for name in required if name not in policy]
    if missing:
        raise KeyError(f"Workflow catheter observation is missing {missing}; got {sorted(policy)}")
    for name, width in zip(STATE_KEYS, STATE_WIDTHS, strict=True):
        actual = policy[name].shape[-1]
        if actual != width:
            raise ValueError(f"expected catheter {name} width {width}, got {actual}")
    return {
        "main_images": _to_rgb(policy["fluoroscopy_rgb"]),
        # Kept for schema parity with multi-camera workflows; a C-arm has no
        # second view to offer.
        "extra_view_images": None,
        "states": torch.cat([policy[name] for name in STATE_KEYS], dim=-1),
        "task_descriptions": [task_description] * num_envs,
    }


def convert_workflow_obs_to_gr00t(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert the bridge schema to the GR00T N1.5 catheter modality contract."""
    import torch

    main = env_obs["main_images"]
    states = env_obs["states"]
    if not all(isinstance(value, torch.Tensor) for value in (main, states)):
        raise TypeError("Workflow catheter images and states must be torch tensors")
    if states.shape[-1] != STATE_DIM:
        raise ValueError(f"expected catheter state width {STATE_DIM}, got {states.shape[-1]}")
    state = states.unsqueeze(1).cpu().numpy()
    return {
        "video.fluoroscopy_view": main.unsqueeze(1).cpu().numpy(),
        "state.catheter": state[:, :, 0:4],
        "state.tip_pose": state[:, :, 4:10],
        "state.navigation": state[:, :, 10:15],
        "annotation.human.task_description": env_obs["task_descriptions"],
    }


def convert_gr00t_to_workflow_action(action_chunk: dict[str, Any], chunk_size: int = 1) -> np.ndarray:
    """Map GR00T's catheter and C-arm heads into Workflow's 4-DoF action.

    No padding: the concatenation is already the full action space, so an
    unexpected width is an error rather than something to pad around.
    """
    missing = [key for key in ACTION_KEYS if key not in action_chunk]
    if missing:
        raise KeyError(f"GR00T catheter action is missing {missing}; got {sorted(action_chunk)}")
    controlled = np.concatenate([np.asarray(action_chunk[key])[:, :chunk_size, :] for key in ACTION_KEYS], axis=-1)
    if controlled.shape[-1] != ACTION_DIM:
        raise ValueError(f"expected {ACTION_DIM} catheter/C-arm channels, got {controlled.shape[-1]}")
    return controlled


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

    class WorkflowCatheterEnv(IsaaclabBaseEnv):
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
            image = policy.get("fluoroscopy_rgb")
            return None if image is None else _to_rgb(image[0]).cpu().numpy()

    return WorkflowCatheterEnv


def register() -> None:
    """Register catheter conversion, model loading, and environment factories."""
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
    logger.info("registered Workflow catheter RL tasks: %s, %s", TRAIN_TASK_ID, EVAL_TASK_ID)

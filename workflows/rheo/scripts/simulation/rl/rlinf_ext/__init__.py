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

import logging
from typing import Any

import cv2
import numpy as np
import torch

logger = logging.getLogger(__name__)

_registered = False


def register() -> None:
    """Register i4h extensions into RLinf.

    This function is called automatically by RLinf's Worker._load_user_extensions()
    when RLINF_EXT_MODULE=rlinf_ext is set in the environment.

    It performs the following registrations:
    1. Imports i4h's IsaacLab task packages (triggers gym.register calls)
    2. Registers task IDs into RLinf's REGISTER_ISAACLAB_ENVS map
    3. Registers GR00T obs/action converters for dex3
    4. Registers GR00T data config for new_embodiment
    5. Monkeypatches RLinf's get_model to support new_embodiment
    """
    global _registered
    if _registered:
        return
    _registered = True

    logger.info("rlinf_ext: Registering i4h extensions...")

    _register_gr00t_converters()

    _register_gr00t_data_config()

    _patch_gr00t_get_model()

    _register_isaaclab_envs()

    logger.info("rlinf_ext: Registration complete.")


def _register_isaaclab_envs() -> None:
    """Register i4h task IDs into RLinf's REGISTER_ISAACLAB_ENVS map."""
    from rlinf.envs.isaaclab import REGISTER_ISAACLAB_ENVS

    # Use factory function to get the class with proper inheritance
    IsaaclabG129Dx3Env = _get_g129_dex3_env_class()

    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Assemble-Trocar-G129-Dex3-Joint", IsaaclabG129Dx3Env)
    REGISTER_ISAACLAB_ENVS.setdefault("Isaac-Assemble-Trocar-G129-Dex3-Joint-Eval", IsaaclabG129Dx3Env)
    logger.debug(f"rlinf_ext: Registered ISAACLAB_ENVS: {list(REGISTER_ISAACLAB_ENVS.keys())}")


def _register_gr00t_converters() -> None:
    """Register GR00T obs/action converters for dex3."""
    from rlinf.models.embodiment.gr00t import simulation_io

    simulation_io.OBS_CONVERSION.setdefault("dex3", _convert_dex3_obs_to_gr00t_format)
    simulation_io.ACTION_CONVERSION.setdefault("dex3", _convert_to_dex3_action)
    logger.debug("rlinf_ext: Registered dex3 obs/action converters")


def _register_gr00t_data_config() -> None:
    """Register GR00T data config for new_embodiment (UnitreeG1Sim)."""
    # Import i4h's data config which adds to DATA_CONFIG_MAP
    import policy.gr00t_config  # noqa: F401

    logger.debug("rlinf_ext: Registered UnitreeG1SimDataConfig")


# ---------------------------------------------------------------------------
# IsaacLab env wrapper for G129 + Dex3
# ---------------------------------------------------------------------------


def _get_g129_dex3_env_class():
    """Factory function to create IsaaclabG129Dx3Env class with proper inheritance.

    This delays the import of IsaaclabBaseEnv until the class is actually needed,
    avoiding import-time side effects in worker processes that don't use IsaacLab.
    """

    from rlinf.envs.isaaclab.isaaclab_env import IsaaclabBaseEnv

    class IsaaclabG129Dx3Env(IsaaclabBaseEnv):
        """Env wrapper for G1 (29DoF) + Dex3 tasks."""

        def __init__(self, cfg, num_envs, seed_offset, total_num_processes, worker_info):
            super().__init__(cfg, num_envs, seed_offset, total_num_processes, worker_info)

        def _make_env_function(self):
            def make_env_isaaclab():
                from isaaclab.app import AppLauncher

                sim_app = AppLauncher(headless=True, enable_cameras=True).app
                import gymnasium as gym
                import simulation.tasks.assemble_trocar  # noqa: F401 - triggers gym.register()
                from isaaclab_tasks.utils import load_cfg_from_registry

                isaac_env_cfg = load_cfg_from_registry(self.isaaclab_env_id, "env_cfg_entry_point")
                isaac_env_cfg.scene.num_envs = self.cfg.init_params.num_envs

                env = gym.make(self.isaaclab_env_id, cfg=isaac_env_cfg, render_mode="rgb_array").unwrapped
                return env, sim_app

            return make_env_isaaclab

        def _wrap_obs(self, obs):
            left_wrist = obs["camera_images"]["left_wrist_camera"]
            right_wrist = obs["camera_images"]["right_wrist_camera"]
            front = obs["camera_images"]["front_camera"]

            dex3_states = obs["policy"]["robot_dex3_joint_state"]  # (B, 14)
            g129_shoulder_states = obs["policy"]["robot_joint_state"][:, 15:29]  # (B, 14)
            states = torch.concatenate([g129_shoulder_states, dex3_states], dim=-1)  # (B, 28)

            task_descriptions = [self.task_description] * self.num_envs
            extra_view_images = torch.stack([left_wrist, right_wrist], dim=1)  # (B, 2, H, W, C)

            return {
                "main_images": front,
                "extra_view_images": extra_view_images,
                "states": states,
                "task_descriptions": task_descriptions,
            }

        def add_image(self, obs):
            """Create a grid of images for video logging."""
            imgs = obs["camera_images"]["front_camera"].cpu().numpy()
            num_envs = imgs.shape[0]

            grid_cols = int(np.ceil(np.sqrt(num_envs)))
            grid_rows = int(np.ceil(num_envs / grid_cols))
            img_h, img_w = imgs.shape[1:3]

            grid_img = np.zeros((grid_rows * img_h, grid_cols * img_w, 3), dtype=np.uint8)

            for idx in range(num_envs):
                row, col = idx // grid_cols, idx % grid_cols
                y0, x0 = row * img_h, col * img_w
                grid_img[y0 : y0 + img_h, x0 : x0 + img_w] = imgs[idx]
                cv2.putText(
                    grid_img,
                    f"Env {idx}",
                    (x0 + 10, y0 + 30),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.8,
                    (255, 255, 255),
                    2,
                )

            return grid_img

    return IsaaclabG129Dx3Env


# ---------------------------------------------------------------------------
# GR00T obs/action converters
# ---------------------------------------------------------------------------


def _convert_dex3_obs_to_gr00t_format(env_obs: dict[str, Any]) -> dict[str, Any]:
    """Convert RLinf env observations into the dict expected by GR00T transforms.

    Expected input schema comes from IsaaclabG129Dx3Env._wrap_obs():
      - main_images: (B, H, W, C) torch tensor
      - extra_view_images: (B, 2, H, W, C) torch tensor [left_wrist, right_wrist]
      - states: (B, 28) torch tensor [left_arm(7), right_arm(7), left_hand(7), right_hand(7)]
      - task_descriptions: list[str] length B
    """
    import torch

    main = env_obs["main_images"]
    extra = env_obs["extra_view_images"]
    states = env_obs["states"]

    if isinstance(main, torch.Tensor):
        # (B, H, W, C) -> (B, T=1, H, W, C)
        room_view = main.unsqueeze(1).cpu().numpy()
        left_wrist = extra[:, 0].unsqueeze(1).cpu().numpy()
        right_wrist = extra[:, 1].unsqueeze(1).cpu().numpy()
        st = states.unsqueeze(1).cpu().numpy()
    else:
        raise TypeError(f"Expected torch.Tensor observations, got {type(main)=}")

    return {
        "video.left_wrist_view": left_wrist,
        "video.right_wrist_view": right_wrist,
        "video.room_view": room_view,
        "state.left_arm": st[:, :, :7],
        "state.right_arm": st[:, :, 7:14],
        "state.left_hand": st[:, :, 14:21],
        "state.right_hand": st[:, :, 21:],
        "annotation.human.action.task_description": env_obs["task_descriptions"],
    }


def _convert_to_dex3_action(action_chunk: dict[str, Any], chunk_size: int = 1) -> Any:
    """Convert GR00T action dict into an action tensor for the IsaacLab env.

    Mirrors the padding behavior:
    - concatenate all action parts along last dim
    - pad 15 zeros at the *front* to align with the full robot joint action space
    """
    import numpy as np

    parts = [v[:, :chunk_size, :] for v in action_chunk.values()]
    action_concat = np.concatenate(parts, axis=-1)
    return np.pad(
        action_concat,
        ((0, 0), (0, 0), (15, 0)),
        mode="constant",
        constant_values=0,
    )


# ---------------------------------------------------------------------------
# Monkeypatch RLinf's get_model to support new_embodiment
# ---------------------------------------------------------------------------


def _patch_embodiment_tags() -> None:
    """Add NEW_EMBODIMENT to RLinf's EmbodimentTag enum and mapping."""
    from rlinf.models.embodiment.gr00t import embodiment_tags

    # Check if NEW_EMBODIMENT already exists
    if not hasattr(embodiment_tags.EmbodimentTag, "NEW_EMBODIMENT"):
        from enum import Enum

        existing_members = {e.name: e.value for e in embodiment_tags.EmbodimentTag}
        existing_members["NEW_EMBODIMENT"] = "new_embodiment"
        NewEmbodimentTag = Enum("EmbodimentTag", existing_members)

        # Replace the old enum with the new one
        embodiment_tags.EmbodimentTag = NewEmbodimentTag

    # Add to mapping if not present
    if "new_embodiment" not in embodiment_tags.EMBODIMENT_TAG_MAPPING:
        embodiment_tags.EMBODIMENT_TAG_MAPPING["new_embodiment"] = 31


def _patch_gr00t_get_model() -> None:
    """Monkeypatch RLinf's GR00T get_model to support new_embodiment."""
    # First, ensure NEW_EMBODIMENT is in the enum
    _patch_embodiment_tags()

    import rlinf.models.embodiment.gr00t as rlinf_gr00t_mod

    original_get_model = rlinf_gr00t_mod.get_model

    def patched_get_model(cfg, torch_dtype=None):  # type: ignore[no-redef]
        import torch

        if torch_dtype is None:
            torch_dtype = torch.bfloat16

        # If not new_embodiment, use original logic
        if cfg.embodiment_tag != "new_embodiment":
            return original_get_model(cfg, torch_dtype=torch_dtype)

        # Handle new_embodiment: use i4h's UnitreeG1SimDataConfig
        from pathlib import Path

        from gr00t.experiment.data_config import load_data_config
        from rlinf.models.embodiment.gr00t.gr00t_action_model import GR00T_N1_5_ForRLActionPrediction
        from rlinf.models.embodiment.gr00t.utils import replace_dropout_with_identity
        from rlinf.utils.patcher import Patcher

        # Apply RLinf's standard EmbodimentTag patches
        Patcher.clear()
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EmbodimentTag",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EmbodimentTag",
        )
        Patcher.add_patch(
            "gr00t.data.embodiment_tags.EMBODIMENT_TAG_MAPPING",
            "rlinf.models.embodiment.gr00t.embodiment_tags.EMBODIMENT_TAG_MAPPING",
        )
        Patcher.apply()

        # Load i4h's data config
        data_config = load_data_config("policy.gr00t_config:UnitreeG1SimDataConfig")
        modality_config = data_config.modality_config()
        modality_transform = data_config.transform()

        model_path = Path(cfg.model_path)
        if not model_path.exists():
            raise FileNotFoundError(f"Model path does not exist: {model_path}")

        model = GR00T_N1_5_ForRLActionPrediction.from_pretrained(
            model_path,
            torch_dtype=torch_dtype,
            embodiment_tag=cfg.embodiment_tag,
            modality_config=modality_config,
            modality_transform=modality_transform,
            denoising_steps=cfg.denoising_steps,
            output_action_chunks=cfg.num_action_chunks,
            obs_converter_type=cfg.obs_converter_type,
            tune_visual=False,
            tune_llm=False,
            rl_head_config=cfg.rl_head_config,
        )
        model.to(torch_dtype)
        if cfg.rl_head_config.add_value_head:
            model.action_head.value_head._init_weights()
        if cfg.rl_head_config.disable_dropout:
            replace_dropout_with_identity(model)

        logger.debug("rlinf_ext: Loaded GR00T model with new_embodiment")
        return model

    rlinf_gr00t_mod.get_model = patched_get_model  # type: ignore[assignment]
    logger.debug("rlinf_ext: Patched get_model for new_embodiment support")

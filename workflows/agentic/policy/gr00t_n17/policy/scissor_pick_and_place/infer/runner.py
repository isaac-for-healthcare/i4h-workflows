# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import importlib.util
import logging
import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from common.config import get_robot_config

logger = logging.getLogger(__name__)
_SO101_CONFIG = get_robot_config("so101")
_ROBOT_ACTION_DIM = _SO101_CONFIG.action_dim
_ROBOT_ARM_JOINT_COUNT = _SO101_CONFIG.arm_joint_count or _SO101_CONFIG.body_joint_count


@dataclass
class RunnerConfig:
    model_path: str
    task_description: str
    embodiment_tag: str = "new_embodiment"
    device: str = "cuda"
    trt_engine_path: Optional[str] = None
    """Directory containing pre-built TRT engines. If set, monkey-patches the
    policy's forward passes with TRT for ~60x speedup. Build with
    ``workflows/soarm/policy/scripts/build_trt_engines.sh``."""
    trt_mode: str = "n17_full_pipeline"
    """TRT acceleration scope. One of ``n17_full_pipeline``, ``vit_llm_only``,
    ``action_head``, or ``dit_only``."""


class GR00TPolicyRunner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self._policy = None
        self._language_key: str | None = None

    def ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(
                f"Model path not found: {self.cfg.model_path}. "
                "Pass --model-path to a local checkpoint or pre-download from Hugging Face."
            )

        # Register the SO-ARM modality config under the chosen embodiment tag
        # so the GR00T processor knows the expected video/state/action shapes.
        # The registration is idempotent — `_ensure_soarm_config_registered`
        # short-circuits if already loaded.
        _ensure_soarm_config_registered()

        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        logger.info("Loading GR00T N1.7 policy from %s ...", self.cfg.model_path)
        t0 = time.monotonic()
        self._policy = Gr00tPolicy(
            embodiment_tag=EmbodimentTag.resolve(self.cfg.embodiment_tag),
            model_path=self.cfg.model_path,
            device=self.cfg.device,
        )
        self._language_key = self._policy.language_key
        logger.info(
            "GR00T policy ready in %.1fs (language_key=%r)",
            time.monotonic() - t0,
            self._language_key,
        )

        if self.cfg.trt_engine_path:
            self._setup_trt(self._policy, self.cfg.trt_engine_path, self.cfg.trt_mode)

    @staticmethod
    def _setup_trt(policy, trt_engine_path: str, mode: str) -> None:
        """Monkey-patch the policy's forward passes with TRT engines.

        Loads upstream ``trt_model_forward.setup_tensorrt_engines`` from the
        vendored Isaac-GR00T checkout. Matches so_arm_starter's runners.py.
        """
        groot_root = Path(__file__).resolve().parents[5] / "third_party" / "Isaac-GR00T-1.7"
        trt_fwd = groot_root / "scripts" / "deployment" / "trt_model_forward.py"

        spec = importlib.util.spec_from_file_location("trt_model_forward", trt_fwd)
        if spec is None or spec.loader is None:
            raise ImportError(f"Cannot find {trt_fwd}")
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)

        logger.info("Setting up TensorRT engines from %s (mode=%s) ...", trt_engine_path, mode)
        mod.setup_tensorrt_engines(policy, trt_engine_path, mode=mode)
        logger.info("TensorRT engines loaded successfully.")

    def infer(self, observation: dict, num_steps: Optional[int] = None) -> np.ndarray:
        self.ensure_loaded()
        room = np.asarray(observation["room"], dtype=np.uint8)
        wrist = np.asarray(observation["wrist"], dtype=np.uint8)
        joint_pos = np.asarray(observation["joint_positions"], dtype=np.float32)

        obs_for_policy = {
            "video": {
                "room": room[np.newaxis, np.newaxis],
                "wrist": wrist[np.newaxis, np.newaxis],
            },
            "state": {
                "single_arm": joint_pos[:_ROBOT_ARM_JOINT_COUNT][np.newaxis, np.newaxis],
                "gripper": joint_pos[_ROBOT_ARM_JOINT_COUNT:_ROBOT_ACTION_DIM][np.newaxis, np.newaxis],
            },
            "language": {
                self._language_key: [[self.cfg.task_description]],
            },
        }

        action_dict, _info = self._policy.get_action(obs_for_policy)
        single_arm = _action_value(action_dict, "single_arm")
        gripper = _action_value(action_dict, "gripper")

        if isinstance(single_arm, np.ndarray):
            single_arm = torch.from_numpy(single_arm)
        if isinstance(gripper, np.ndarray):
            gripper = torch.from_numpy(gripper)

        # action shapes are (B, T, D); drop batch dim
        if single_arm.dim() == 3:
            single_arm = single_arm.squeeze(0)
        if gripper.dim() == 3:
            gripper = gripper.squeeze(0)
        if single_arm.dim() == 1:
            single_arm = single_arm.unsqueeze(0)
        if gripper.dim() == 1:
            gripper = gripper.unsqueeze(-1)

        # The bootstrap checkpoint only commands a single gripper channel, so
        # broadcast it across every finger joint when the robot has more than
        # one (e.g. parallel grippers).
        finger_count = _ROBOT_ACTION_DIM - _ROBOT_ARM_JOINT_COUNT
        if gripper.shape[-1] < finger_count:
            gripper = gripper[..., :1].repeat_interleave(finger_count, dim=-1)
        elif gripper.shape[-1] > finger_count:
            gripper = gripper[..., :finger_count]

        action = torch.cat([single_arm, gripper], dim=-1).detach().cpu().numpy().astype(np.float64)
        out_steps = min(num_steps if num_steps is not None else action.shape[0], action.shape[0])
        return np.asarray(action[:out_steps], dtype=np.float64)


def _action_value(action_dict: dict, key: str):
    # n1.7 returns keys directly: "single_arm", "gripper"; tolerate older "action.<key>" form too.
    for candidate in (key, f"action.{key}", f"state.{key}"):
        value = action_dict.get(candidate)
        if value is not None:
            return value
    raise ValueError(f"Could not find GR00T action for {key}. Available keys: {list(action_dict.keys())}")


def _ensure_soarm_config_registered() -> None:
    """Register the SO-ARM modality config exactly once.

    The GR00T processor maps an embodiment tag to a ModalityConfig (video keys,
    state keys, action horizon). We mirror the registration done at training
    time so that the policy created here knows what observations to expect.
    """
    from gr00t.configs.data.embodiment_configs import MODALITY_CONFIGS
    from gr00t.data.embodiment_tags import EmbodimentTag

    if EmbodimentTag.NEW_EMBODIMENT.value in MODALITY_CONFIGS:
        return
    # Importing soarm_config triggers register_modality_config().
    from policy.scissor_pick_and_place import config as soarm_config  # noqa: F401

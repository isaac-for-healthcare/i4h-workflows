# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
import time
from contextlib import contextmanager
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from common.config import get_robot_config

logger = logging.getLogger(__name__)
_SO101_CONFIG = get_robot_config("so101")
_ROBOT_ACTION_DIM = _SO101_CONFIG.action_dim
_ROBOT_ARM_JOINT_COUNT = _SO101_CONFIG.arm_joint_count or _SO101_CONFIG.body_joint_count
_ROBOT_LEROBOT_JOINT_POS_LIMIT_RANGE = _SO101_CONFIG.lerobot_joint_pos_limit_range


@dataclass
class RunnerConfig:
    model_path: str
    task_description: str
    data_config: str = "so100_dualcam"
    embodiment_tag: str = "new_embodiment"
    denoising_steps: int = 4
    action_head_future_tokens: int | None = 0
    trt_engine_path: str | None = None


class GR00TPolicyRunner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self._policy = None
        self._modality_keys = ("single_arm", "gripper")

    def ensure_loaded(self) -> None:
        if self._policy is not None:
            return
        if not os.path.exists(self.cfg.model_path):
            raise FileNotFoundError(
                f"Model path not found: {self.cfg.model_path}. "
                "Pass --model-path to a local checkpoint or pre-download from Hugging Face."
            )

        logger.info("Loading GR00T N1.5 policy from %s ...", self.cfg.model_path)
        t0 = time.monotonic()
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        data_config = DATA_CONFIG_MAP[self.cfg.data_config]
        data_config.video_keys = ["video.room", "video.wrist"]
        # The bootstrap checkpoint (nvidia/SO_ARM_Starter_Gr00t) ships a
        # metadata.json sized for the 5+1 SO-ARM. Override it at construction
        # so observations for the current robot are not rejected by the
        # modality transform. After retraining on the current robot, the new
        # checkpoint's metadata matches and this override becomes a no-op.
        with _patched_gr00t_metadata_load():
            self._policy = Gr00tPolicy(
                model_path=self.cfg.model_path,
                modality_config=data_config.modality_config(),
                modality_transform=data_config.transform(),
                embodiment_tag=self.cfg.embodiment_tag,
                denoising_steps=self.cfg.denoising_steps,
                action_head_future_tokens=self.cfg.action_head_future_tokens,
            )
        if self.cfg.trt_engine_path:
            from policy.scissor_pick_and_place.infer.tensorrt_dit import replace_dit_with_tensorrt

            replace_dit_with_tensorrt(self._policy, self.cfg.trt_engine_path)
        logger.info("GR00T policy ready in %.1fs", time.monotonic() - t0)

    def infer(self, observation: dict, num_steps: Optional[int] = None) -> np.ndarray:
        self.ensure_loaded()
        room = observation["room"][np.newaxis, ...]
        wrist = observation["wrist"][np.newaxis, ...]
        joint_pos = np.asarray(observation["joint_positions"], dtype=np.float64)

        obs_for_policy = {
            "video.room": room,
            "video.wrist": wrist,
            "state.single_arm": joint_pos[:_ROBOT_ARM_JOINT_COUNT][np.newaxis, ...],
            "state.gripper": joint_pos[_ROBOT_ARM_JOINT_COUNT:_ROBOT_ACTION_DIM][np.newaxis, ...],
            "annotation.human.task_description": self.cfg.task_description,
        }
        action_chunk = self._policy.get_action(obs_for_policy)
        single_arm = _action_value(action_chunk, "single_arm")
        gripper = _action_value(action_chunk, "gripper")

        if isinstance(single_arm, np.ndarray):
            single_arm = torch.from_numpy(single_arm)
        if isinstance(gripper, np.ndarray):
            gripper = torch.from_numpy(gripper)

        if single_arm.dim() > 2:
            single_arm = single_arm.squeeze(0)
        if gripper.dim() > 2:
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


def _action_value(action_chunk: dict, key: str):
    for candidate in (f"action.{key}", f"state.{key}", key):
        value = action_chunk.get(candidate)
        if value is not None:
            return value
    raise ValueError(f"Could not find GR00T action for {key}. Available keys: {list(action_chunk.keys())}")


def _build_runtime_metadata_dict() -> dict:
    arm_range = _ROBOT_LEROBOT_JOINT_POS_LIMIT_RANGE[:_ROBOT_ARM_JOINT_COUNT]
    gripper_range = _ROBOT_LEROBOT_JOINT_POS_LIMIT_RANGE[_ROBOT_ARM_JOINT_COUNT:_ROBOT_ACTION_DIM]

    def _stats(ranges):
        mins = [float(lo) for lo, _ in ranges]
        maxs = [float(hi) for _, hi in ranges]
        means = [(lo + hi) / 2.0 for lo, hi in ranges]
        stds = [(hi - lo) / 4.0 for lo, hi in ranges]
        q01 = [lo + 0.05 * (hi - lo) for lo, hi in ranges]
        q99 = [hi - 0.05 * (hi - lo) for lo, hi in ranges]
        return {"min": mins, "max": maxs, "mean": means, "std": stds, "q01": q01, "q99": q99}

    arm_stats, gripper_stats = _stats(arm_range), _stats(gripper_range)
    block = {
        "single_arm": {"absolute": True, "rotation_type": None, "shape": [len(arm_range)], "continuous": True},
        "gripper": {"absolute": True, "rotation_type": None, "shape": [len(gripper_range)], "continuous": True},
    }
    return {
        "statistics": {
            "state": {"single_arm": arm_stats, "gripper": gripper_stats},
            "action": {"single_arm": arm_stats, "gripper": gripper_stats},
        },
        "modalities": {
            "video": {
                "room": {"resolution": [640, 480], "channels": 3, "fps": 30.0},
                "wrist": {"resolution": [640, 480], "channels": 3, "fps": 30.0},
            },
            "state": block,
            "action": block,
        },
        "embodiment_tag": "new_embodiment",
    }


@contextmanager
def _patched_gr00t_metadata_load():
    """Only patch when the on-disk metadata shape disagrees with SO-ARM.

    For SO-ARM 101 + nvidia/SO_ARM_Starter_Gr00t the on-disk metadata is
    correct (single_arm=[5], gripper=[1]) and ships real trained statistics
    that the policy normalizer depends on. Replacing those stats with
    synthetic uniform-range stats produces visibly wrong arm motion. So we
    let `Gr00tPolicy._load_metadata` run normally first, then check the
    shape; only if the checkpoint's metadata is sized for a different
    embodiment do we replace it with a runtime-built dict.
    """
    from gr00t.data.schema import DatasetMetadata
    from gr00t.model.policy import Gr00tPolicy

    original = Gr00tPolicy._load_metadata
    expected_arm = _ROBOT_ARM_JOINT_COUNT
    expected_gripper = _ROBOT_ACTION_DIM - _ROBOT_ARM_JOINT_COUNT

    def _patched(self_policy, exp_cfg_dir):
        original(self_policy, exp_cfg_dir)
        if _metadata_matches_robot(getattr(self_policy, "metadata", None), expected_arm, expected_gripper):
            logger.info(
                "GR00T metadata matches SO-ARM (single_arm=[%s], gripper=[%s]); using checkpoint metadata.",
                expected_arm,
                expected_gripper,
            )
            return
        metadata = DatasetMetadata.model_validate(_build_runtime_metadata_dict())
        self_policy._modality_transform.set_metadata(metadata)
        self_policy.metadata = metadata
        logger.info(
            "GR00T metadata override active: state.single_arm=[%s], state.gripper=[%s] "
            "(checkpoint metadata shape did not match SO-ARM).",
            expected_arm,
            expected_gripper,
        )

    Gr00tPolicy._load_metadata = _patched
    try:
        yield
    finally:
        Gr00tPolicy._load_metadata = original


def _metadata_matches_robot(metadata, expected_arm: int, expected_gripper: int) -> bool:
    if metadata is None:
        return False
    modalities = getattr(metadata, "modalities", None)
    if modalities is None:
        return False
    state = getattr(modalities, "state", None) or {}

    def _shape_value(block):
        if block is None:
            return None
        shape = getattr(block, "shape", None) if not isinstance(block, dict) else block.get("shape")
        if shape is None:
            return None
        try:
            return int(shape[0])
        except (TypeError, IndexError, ValueError):
            return None

    single_arm_shape = _shape_value(
        state.get("single_arm") if isinstance(state, dict) else getattr(state, "single_arm", None)
    )
    gripper_shape = _shape_value(state.get("gripper") if isinstance(state, dict) else getattr(state, "gripper", None))
    return single_arm_shape == expected_arm and gripper_shape == expected_gripper

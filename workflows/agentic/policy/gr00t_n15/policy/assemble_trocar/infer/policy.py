# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("policy")


class AssembleTrocarPolicy:
    """GR00T N1.5 policy adapter for the 43-D Assemble Trocar Arena task."""

    def __init__(
        self,
        *,
        model_path: str | None,
        model_repo: str | None,
        model_revision: str | None,
        task_description: str,
        device: str = "cuda",
        action_head_future_tokens: int | None = 32,
        trt_engine_path: str | None = None,
    ) -> None:
        self.device = device
        self.task_description = task_description
        self.action_head_future_tokens = action_head_future_tokens
        self.trt_engine_path = trt_engine_path
        self.model_path = _resolve_model_path(model_path, model_repo, model_revision)
        self.policy = self._load_policy()

    def _load_policy(self) -> Any:
        import policy.assemble_trocar.infer.data_config  # noqa: F401
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values

        data_config = DATA_CONFIG_MAP["unitree_g1_assemble_trocar"]
        self._action_keys = tuple(data_config.action_keys)

        class AssembleTrocarRLPolicy(Gr00tPolicy):
            """GR00T N1.5 wrapper with RL checkpoint compatibility applied in-process."""

            def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
                is_batch = self._check_state_is_batched(observations)
                if not is_batch:
                    observations = unsqueeze_dict_values(observations)

                normalized_input = self.apply_transforms(observations)
                normalized_input = _prepare_rl_normalized_input(normalized_input)
                normalized_action = self._get_action_from_normalized_input(normalized_input)
                unnormalized_action = self._get_unnormalized_action(normalized_action)

                if not is_batch:
                    unnormalized_action = squeeze_dict_values(unnormalized_action)
                return unnormalized_action

            def _load_model(self, model_path):
                super()._load_model(model_path)
                _replace_dropout_with_identity(self.model)

        policy = AssembleTrocarRLPolicy(
            model_path=str(self.model_path),
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag="new_embodiment",
            device=self.device,
            action_head_future_tokens=self.action_head_future_tokens,
        )
        if self.trt_engine_path:
            from policy.assemble_trocar.infer.tensorrt_dit import replace_dit_with_tensorrt

            device_idx = 0
            if ":" in self.device:
                device_idx = int(self.device.split(":")[-1])
            replace_dit_with_tensorrt(policy, self.trt_engine_path, device=device_idx)
        return policy

    def get_action(self, observation: dict[str, Any]) -> np.ndarray:
        obs = self._format_observation(observation)
        action_dict = self.policy.get_action(obs)
        action_chunk = np.concatenate(
            [np.atleast_1d(action_dict[key]) for key in self._action_keys],
            axis=-1,
        )
        if action_chunk.ndim == 2:
            action_chunk = action_chunk[np.newaxis, :, :]
        if action_chunk.shape[-1] == 28:
            zero_base = np.zeros((action_chunk.shape[0], action_chunk.shape[1], 15), dtype=action_chunk.dtype)
            action_chunk = np.concatenate([zero_base, action_chunk], axis=-1)
        return action_chunk[0].astype(np.float32)

    def _format_observation(self, observation: dict[str, Any]) -> dict[str, Any]:
        body, hands = _split_joint_positions(observation)
        frames = observation["frames"]
        return {
            "video.room_view": _image_array(frames["front"]),
            "video.left_wrist_view": _image_array(frames["left_wrist"]),
            "video.right_wrist_view": _image_array(frames["right_wrist"]),
            "state.left_arm": body[:, :, 15:22].astype(np.float32, copy=False),
            "state.right_arm": body[:, :, 22:29].astype(np.float32, copy=False),
            "state.left_hand": hands[:, :, :7].astype(np.float32, copy=False),
            "state.right_hand": hands[:, :, 7:14].astype(np.float32, copy=False),
            "annotation.human.task_description": [self.task_description],
        }


def _image_array(image: np.ndarray) -> np.ndarray:
    array = np.asarray(image)
    if array.dtype != np.uint8:
        array = np.clip(array, 0, 255).astype(np.uint8)
    return array.reshape(1, 1, *array.shape)


def _split_joint_positions(observation: dict[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    """Validate the 43-D joint_positions vector and split it into the body
    (29-D, first part) and hand (14-D, dex3) chunks both formatters share."""
    joint_positions = np.asarray(observation["joint_positions"], dtype=np.float32)
    if joint_positions.shape[-1] != 43:
        raise ValueError(f"Assemble Trocar expects 43 joint positions, received {joint_positions.shape[-1]}")
    body = joint_positions[:29].reshape(1, 1, -1)
    hands = joint_positions[29:].reshape(1, 1, -1)
    return body, hands


def _prepare_rl_normalized_input(normalized_input: dict[str, Any]) -> dict[str, Any]:
    prepared = dict(normalized_input)
    for key, value in prepared.items():
        if torch.is_tensor(value) and value.dtype == torch.float32:
            prepared[key] = value.to(torch.bfloat16)
    for key in ("eagle_input_ids", "eagle_attention_mask"):
        value = prepared.get(key)
        if torch.is_tensor(value) and value.shape[-1] < 850:
            prepared[key] = F.pad(value, pad=(0, 850 - value.shape[-1]), mode="constant", value=0)
    return prepared


def _replace_dropout_with_identity(model: nn.Module) -> None:
    modules = dict(model.named_modules())
    for name, module in list(modules.items()):
        if isinstance(module, (nn.Dropout, nn.Dropout1d, nn.Dropout2d, nn.Dropout3d, nn.AlphaDropout)):
            parent_name, _, child_name = name.rpartition(".")
            parent = modules[parent_name] if parent_name else model
            setattr(parent, child_name, nn.Identity())


def _resolve_model_path(model_path: str | None, model_repo: str | None, model_revision: str | None) -> Path:
    if model_path:
        path = Path(model_path).expanduser()
        if not path.exists():
            raise FileNotFoundError(f"Assemble Trocar --model-path not found: {path}")
        logger.info("Using Assemble Trocar model from --model-path: %s", path)
        return path

    if not model_repo:
        raise FileNotFoundError("No Assemble Trocar model repo configured.")

    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("RHEO_MODEL_CACHE", os.path.expanduser("~/.cache/rheo_models"))
    os.makedirs(cache_dir, exist_ok=True)
    logger.info(
        "Pulling Assemble Trocar model repo %s revision=%s into %s",
        model_repo,
        model_revision or "<default>",
        cache_dir,
    )
    return Path(snapshot_download(repo_id=model_repo, revision=model_revision, cache_dir=cache_dir))

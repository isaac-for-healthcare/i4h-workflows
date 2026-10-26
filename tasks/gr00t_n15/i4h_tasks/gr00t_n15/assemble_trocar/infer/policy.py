# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

import torch
import torch.nn as nn
import torch.nn.functional as F

logger = logging.getLogger("policy")


def load_assemble_trocar_policy(
    model_path: str | Path,
    *,
    device: str = "cuda",
    action_head_future_tokens: int | None = 32,
) -> Any:
    """The Trocar checkpoint as a GR00T N1.5 ``Gr00tPolicy`` taking model-format observations.

    The RL post-trained checkpoint needs bf16 inputs, Eagle tokens padded to 850 and no
    dropout at inference; this policy applies all three.
    """
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy, squeeze_dict_values, unsqueeze_dict_values

    import i4h_tasks.gr00t_n15.assemble_trocar.infer.data_config  # noqa: F401

    data_config = DATA_CONFIG_MAP["unitree_g1_assemble_trocar"]

    class AssembleTrocarRLPolicy(Gr00tPolicy):
        """GR00T N1.5 policy with the task checkpoint's required preprocessing."""

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

    return AssembleTrocarRLPolicy(
        model_path=str(model_path),
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag="new_embodiment",
        device=device,
        action_head_future_tokens=action_head_future_tokens,
    )


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


def resolve_model_path(model_path: str | None, model_repo: str | None, model_revision: str | None) -> Path:
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

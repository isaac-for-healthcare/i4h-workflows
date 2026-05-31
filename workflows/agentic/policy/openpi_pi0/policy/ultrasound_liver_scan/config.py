# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""openpi training-config registration for the ultrasound liver-scan PI0 model.

Lifted from ``workflows/robotic_ultrasound/scripts/policy/pi0/config.py`` —
the same dataset / model layout the released checkpoints expect.
"""

from __future__ import annotations

from openpi.models.pi0 import Pi0Config
from openpi.training.config import DataConfig, TrainConfig
from openpi.training.weight_loaders import CheckpointWeightLoader
from policy.ultrasound_liver_scan.utils import LeRobotDataConfig

_CONFIG_REGISTRY: dict = {}


def register_config(name: str):
    def _register(config_fn):
        _CONFIG_REGISTRY[name] = config_fn
        return config_fn

    return _register


def get_config(name: str, repo_id: str, exp_name: str | None = None) -> TrainConfig:
    if name not in _CONFIG_REGISTRY:
        raise ValueError(f"Config '{name}' not found. Available: {list(_CONFIG_REGISTRY.keys())}")
    return _CONFIG_REGISTRY[name](repo_id, exp_name)


@register_config("robotic_ultrasound")
def _robotic_ultrasound(repo_id: str, exp_name: str | None) -> TrainConfig:
    return TrainConfig(
        name="robotic_ultrasound",
        model=Pi0Config(),
        data=LeRobotDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
                # Agentic LeRobot converter writes the action sequence under "action"
                # (singular), not openpi's legacy default "actions".
                action_sequence_keys=("action",),
            ),
        ),
        weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        resume=True,
        exp_name=exp_name,
    )

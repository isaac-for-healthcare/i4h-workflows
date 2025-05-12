# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

from openpi.models.pi0 import Pi0Config
from openpi.training.config import DataConfig, TrainConfig
from openpi.training.weight_loaders import CheckpointWeightLoader
from policy_runner.pi0_policy.utils import LeRobotDataConfig

# Config registry to store all available configurations
_CONFIG_REGISTRY = {}


def register_config(name: str):
    """Decorator to register a configuration function in the registry.

    Args:
        name (str): The name to register this configuration under in the registry.

    Returns:
        callable: The decorated configuration function.

    Example:
        @register_config("my_config")
        def get_my_config():
            return TrainConfig(...)
    """

    def _register(config_fn):
        _CONFIG_REGISTRY[name] = config_fn
        return config_fn

    return _register


def get_config(name: str, repo_id: str, exp_name: str = None) -> TrainConfig:
    """Get a training configuration by name from the registry.

    Args:
        name (str): Name of the configuration to retrieve.
        repo_id (str): Repository ID for the dataset, used for normalization stats.
        exp_name (str, optional): Name for the experiment, used for logging and checkpoints.

    Returns:
        TrainConfig: The requested training configuration.

    Raises:
        ValueError: If the requested configuration name is not found in the registry.
    """
    if name not in _CONFIG_REGISTRY:
        raise ValueError(f"Config '{name}' not found. Available configs: {list(_CONFIG_REGISTRY.keys())}")
    return _CONFIG_REGISTRY[name](repo_id, exp_name)


# Register configurations
@register_config("robotic_ultrasound")
def get_robotic_ultrasound_config(repo_id: str, exp_name: str):
    """Get the full fine-tuning configuration for robotic ultrasound.

    This configuration performs full fine-tuning of the PI0 model for robotic ultrasound tasks.
    Requires significant GPU memory (>70GB) but can achieve better performance on larger datasets.

    Args:
        repo_id (str): Repository ID for the dataset, used for normalization stats.
        exp_name (str): Name for the experiment, used for logging and checkpoints.

    Returns:
        TrainConfig: Training configuration for full fine-tuning.
    """
    return TrainConfig(
        name="robotic_ultrasound",
        model=Pi0Config(),
        data=LeRobotDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(
                local_files_only=True,  # Set to True for local-only datasets.
                prompt_from_task=True,
            ),
        ),
        weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        resume=True,
        exp_name=exp_name,
    )


@register_config("robotic_ultrasound_lora")
def get_robotic_ultrasound_lora_config(repo_id: str, exp_name: str):
    """Get the LoRA fine-tuning configuration for robotic ultrasound.

    This configuration uses Low-Rank Adaptation (LoRA) to fine-tune a subset of model parameters.
    Requires less GPU memory (~22.5GB) while still achieving good results. Recommended for most use cases.

    Args:
        repo_id (str): Repository ID for the dataset, used for normalization stats.
        exp_name (str): Name for the experiment, used for logging and checkpoints.

    Returns:
        TrainConfig: Training configuration for LoRA fine-tuning.
    """
    return TrainConfig(
        name="robotic_ultrasound_lora",
        model=Pi0Config(),
        data=LeRobotDataConfig(
            repo_id=repo_id,
            base_config=DataConfig(
                local_files_only=True,
                prompt_from_task=True,
            ),
        ),
        weight_loader=CheckpointWeightLoader("s3://openpi-assets/checkpoints/pi0_base/params"),
        num_train_steps=30_000,
        freeze_filter=Pi0Config(
            paligemma_variant="gemma_2b_lora", action_expert_variant="gemma_300m_lora"
        ).get_freeze_filter(),
        ema_decay=None,
        resume=True,
        exp_name=exp_name,
    )

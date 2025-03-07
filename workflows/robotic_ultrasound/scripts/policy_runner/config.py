from policy_runner.utils import LeRobotDataConfig

from openpi.models.pi0 import Pi0Config
from openpi.training.config import DataConfig, TrainConfig
from openpi.training.weight_loaders import CheckpointWeightLoader

# Config registry to store all available configurations
_CONFIG_REGISTRY = {}


def register_config(name: str):
    """Decorator to register a configuration function in the registry."""

    def _register(config_fn):
        _CONFIG_REGISTRY[name] = config_fn
        return config_fn

    return _register


def get_config(name: str, repo_id: str, exp_name: str = None) -> TrainConfig:
    """Get a configuration by name from the registry."""
    if name not in _CONFIG_REGISTRY:
        raise ValueError(f"Config '{name}' not found. Available configs: {list(_CONFIG_REGISTRY.keys())}")
    return _CONFIG_REGISTRY[name](repo_id, exp_name)


# Register configurations
@register_config("robotic_ultrasound")
def get_robotic_ultrasound_config(repo_id: str, exp_name: str):
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

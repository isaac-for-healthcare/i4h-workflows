# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingModuleSource=false
from __future__ import annotations

import os
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

_WORKFLOW_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RobotConfig:
    id: str
    body_joint_names: tuple[str, ...]
    hand_joint_names: tuple[str, ...] = ()
    has_locomotion: bool = False
    has_dex_hand: bool = False
    arm_joint_count: int | None = None
    isaaclab_joint_pos_limit_range: tuple[tuple[float, float], ...] | None = None
    lerobot_joint_pos_limit_range: tuple[tuple[float, float], ...] | None = None
    locomanip_embodiment_name: str | None = None
    locomanip_teleop_embodiment_name: str | None = None
    assemble_trocar_embodiment_name: str | None = None
    teleop_devices: tuple[str, ...] = ()
    teleop_device_configs: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def joint_names(self) -> tuple[str, ...]:
        return self.body_joint_names + self.hand_joint_names

    @property
    def body_joint_count(self) -> int:
        return len(self.body_joint_names)

    @property
    def hand_joint_count(self) -> int:
        return len(self.hand_joint_names)

    @property
    def action_dim(self) -> int:
        return len(self.joint_names)


@dataclass(frozen=True)
class PolicyConfig:
    env_id: str
    stack: str | None = None
    infer_module: str | None = None
    train_module: str | None = None
    add_args_attr: str | None = None
    run_attr: str | None = None
    model_repo: str | None = None
    model_revision: str | None = None
    task_description: str | None = None
    data_config: str | None = None
    embodiment_tag: str | None = None
    action_horizon: int | None = None
    action_head_future_tokens: int | None = None
    denoising_steps: int | None = None
    execution_steps: int | None = None
    control_hz: float | None = None
    trt_engine_path: str | None = None
    language_instruction: str | None = None
    policy_joints_config_path: str | None = None
    action_joints_config_path: str | None = None
    state_joints_config_path: str | None = None
    action_chunk_length: int | None = None
    # Optional multi-camera rollout mapping.
    pov_cam_names_sim: list | None = None
    task_mode_name: str | None = None
    repo_id: str | None = None
    image_size: list[int] | None = None
    health_port: int | None = None
    train: dict[str, Any] = field(default_factory=dict)
    action_overrides: dict[str, float] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        routing_fields = {"env_id", "stack", "infer_module", "train_module", "add_args_attr", "run_attr", "train"}
        return {key: value for key, value in asdict(self).items() if key not in routing_fields and value is not None}

    @property
    def required_health_port(self) -> int:
        if self.health_port is None:
            raise ValueError(
                f"policy.health_port is not set for env {self.env_id!r}; "
                f"add it to config/environments/{self.env_id}.yaml"
            )
        return self.health_port

    @property
    def required_model_repo(self) -> str:
        return self._required("model_repo")

    @property
    def required_language_instruction(self) -> str:
        return self._required("language_instruction")

    def _required(self, field: str) -> Any:
        value = getattr(self, field)
        if value is None:
            raise ValueError(f"policy config for {self.env_id} is missing required field {field!r}")
        return value


@dataclass(frozen=True)
class EnvMetadata:
    env_id: str
    description: str
    runtime_module: str


@dataclass(frozen=True)
class PolicyRouting:
    env_id: str
    stack: str
    description: str
    language_description: str
    infer_module: str
    train_module: str | None
    add_args_attr: str = "add_args"
    run_attr: str = "run"


@dataclass(frozen=True)
class ZenohEnvConfig:
    env_id: str
    camera_names: tuple[str, ...]

    @property
    def camera_keys(self) -> dict[str, str]:
        return {name: self.camera_key(name) for name in self.camera_names}

    @property
    def robot_state_key(self) -> str:
        return self.key("robot/state")

    @property
    def robot_command_key(self) -> str:
        return self.key("robot/command")

    def camera_key(self, name: str) -> str:
        return self.key(f"camera/{name}")

    def key(self, suffix: str) -> str:
        return f"i4h/agentic/{self.env_id}/{suffix}"


@lru_cache(maxsize=None)
def _workflow_root() -> Path:
    root = os.environ.get("WORKFLOW_ROOT")
    return Path(root).expanduser().resolve() if root else _WORKFLOW_ROOT


@lru_cache(maxsize=None)
def environment_config_path() -> Path:
    explicit = os.environ.get("ENVIRONMENT_CONFIG")
    path = Path(explicit).expanduser().resolve() if explicit else _workflow_root() / "config" / "environments"
    if explicit and not path.exists():
        raise FileNotFoundError(f"expected environment config at {path}; set ENVIRONMENT_CONFIG to override")
    return path


@lru_cache(maxsize=None)
def _config_data() -> dict[str, Any]:
    if os.environ.get("ENVIRONMENT_CONFIG"):
        return _load_yaml_file(environment_config_path())

    environments = _load_named_yaml_dir(environment_config_path())
    if environments:
        return {"environments": environments}

    return {}


@lru_cache(maxsize=None)
def environment_blocks() -> dict[str, dict[str, Any]]:
    return _config_data().get("environments") or {}


@lru_cache(maxsize=None)
def get_environment_config(env_id: str) -> dict[str, Any]:
    blocks = environment_blocks()
    try:
        return blocks[env_id]
    except KeyError as exc:
        choices = ", ".join(sorted(blocks))
        raise KeyError(f"unknown environment {env_id!r}; available: {choices}") from exc


@lru_cache(maxsize=None)
def get_env_metadata(env_id: str) -> EnvMetadata:
    env_config = get_environment_config(env_id)
    arena = env_config.get("arena")
    if not isinstance(arena, dict):
        raise ValueError(f"environment {env_id!r} must define arena config in config/environments/{env_id}.yaml")
    if "description" not in arena:
        raise ValueError(f"environment {env_id!r} must define arena.description in config/environments/{env_id}.yaml")
    description = str(arena["description"])
    if not description.strip():
        raise ValueError(f"environment {env_id!r} must define non-empty arena.description")
    runtime_module = arena.get("runtime_module") or f"arena.runtimes.{env_id}"
    return EnvMetadata(
        env_id=env_id,
        description=description,
        runtime_module=str(runtime_module),
    )


@lru_cache(maxsize=None)
def get_policy_routing(env_id: str) -> PolicyRouting:
    env_config = get_environment_config(env_id)
    policy = env_config.get("policy")
    if not isinstance(policy, dict):
        raise KeyError(f"environment {env_id!r} does not define policy config")
    stack = policy.get("stack")
    if not stack:
        raise ValueError(f"environment {env_id!r} must define policy.stack in config/environments/{env_id}.yaml")
    metadata = get_env_metadata(env_id)
    train_module = policy["train_module"] if "train_module" in policy else f"policy.{env_id}.train.train"
    return PolicyRouting(
        env_id=env_id,
        stack=str(stack),
        description=metadata.description,
        language_description=str(
            policy.get("language_instruction") or policy.get("task_description") or metadata.description
        ),
        infer_module=str(policy.get("infer_module") or f"policy.{env_id}.infer.infer"),
        train_module=None if train_module is None else str(train_module),
        add_args_attr=str(policy.get("add_args_attr") or "add_args"),
        run_attr=str(policy.get("run_attr") or "run"),
    )


def policy_routings() -> tuple[PolicyRouting, ...]:
    return tuple(get_policy_routing(env_id) for env_id in sorted(environment_blocks()))


def policy_routings_for_stack(stack: str) -> tuple[PolicyRouting, ...]:
    return tuple(routing for routing in policy_routings() if routing.stack == stack)


def policy_stack_for_env(env_id: str) -> str:
    return get_policy_routing(env_id).stack


@lru_cache(maxsize=None)
def _robot_config_dir() -> Path:
    explicit = os.environ.get("ROBOT_CONFIG_DIR")
    return Path(explicit).expanduser().resolve() if explicit else _workflow_root() / "config" / "robots"


@lru_cache(maxsize=None)
def get_robot_config(robot_id: str) -> RobotConfig:
    path = _robot_config_dir() / f"{robot_id}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"unknown robot config {robot_id!r}; expected {path}")
    data = yaml.safe_load(path.read_text()) or {}
    teleop_device_configs = _teleop_device_configs(data.get("teleop_devices"))
    return RobotConfig(
        id=str(data.get("id") or robot_id),
        body_joint_names=tuple(str(name) for name in data.get("body_joint_names", ())),
        hand_joint_names=tuple(str(name) for name in data.get("hand_joint_names", ())),
        has_locomotion=bool(data.get("has_locomotion", False)),
        has_dex_hand=bool(data.get("has_dex_hand", False)),
        arm_joint_count=_optional_int(data.get("arm_joint_count")),
        isaaclab_joint_pos_limit_range=_optional_range(data.get("isaaclab_joint_pos_limit_range")),
        lerobot_joint_pos_limit_range=_optional_range(data.get("lerobot_joint_pos_limit_range")),
        locomanip_embodiment_name=data.get("locomanip_embodiment_name"),
        locomanip_teleop_embodiment_name=data.get("locomanip_teleop_embodiment_name"),
        assemble_trocar_embodiment_name=data.get("assemble_trocar_embodiment_name"),
        teleop_devices=tuple(teleop_device_configs),
        teleop_device_configs=teleop_device_configs,
    )


@lru_cache(maxsize=None)
def get_env_robot_config(env_id: str) -> RobotConfig:
    env_config = get_environment_config(env_id)
    robot_id = (env_config.get("robot") or {}).get("type") or env_config.get("robot_type")
    if not robot_id:
        raise ValueError(f"environment {env_id!r} must define robot.type")
    return get_robot_config(str(robot_id))


@lru_cache(maxsize=None)
def get_policy_config(env_id: str) -> PolicyConfig:
    env_config = get_environment_config(env_id)
    policy = env_config.get("policy")
    if not policy:
        raise KeyError(f"environment {env_id!r} does not define policy config")
    return PolicyConfig(env_id=env_id, **policy)


@lru_cache(maxsize=None)
def get_zenoh_config(env_id: str) -> ZenohEnvConfig:
    env_config = get_environment_config(env_id)
    camera_names = tuple((env_config.get("zenoh") or {}).get("camera_names") or ())
    if not camera_names:
        raise ValueError(f"environment {env_id!r} must define zenoh.camera_names")
    return ZenohEnvConfig(env_id, camera_names)


def _load_named_yaml_dir(path: Path) -> dict[str, dict[str, Any]]:
    if not path.is_dir():
        return {}
    return {item.stem: _load_yaml_file(item) for item in sorted(path.glob("*.yaml"))}


def _load_yaml_file(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text()) or {}


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_range(value: Any) -> tuple[tuple[float, float], ...] | None:
    if value is None:
        return None
    return tuple((float(lo), float(hi)) for lo, hi in value)


def _teleop_device_configs(value: Any) -> dict[str, dict[str, Any]]:
    if value is None:
        return {}
    if isinstance(value, dict):
        return {str(name): dict(config or {}) for name, config in value.items()}
    return {str(name): {} for name in value}

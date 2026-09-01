# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lightweight RL profile loading; safe before a trainer or simulator starts."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from i4h_rl.backend_loader import BACKEND_MODULES


class ProfileError(ValueError):
    """An RL training profile is incomplete or inconsistent."""


@dataclass(frozen=True, slots=True)
class SimulationProfile:
    """Simulator launch facts shared by trainer backends."""

    env_spacing: float
    presets: str
    enable_cameras: bool


@dataclass(frozen=True, slots=True)
class RLProfile:
    schema_version: int
    workflow: str
    scene: str
    trainer: str
    algorithm: str
    adapter_module: str | None
    trainer_config: Path
    train_task_id: str
    eval_task_id: str
    task_description: str
    action_dof: int
    policy_action_dof: int
    cameras: tuple[str, ...]
    state_dof: int
    default_num_envs: int
    default_epochs: int
    simulation: SimulationProfile

    @classmethod
    def load(cls, path: Path) -> RLProfile:
        try:
            raw = yaml.safe_load(path.read_text(encoding="utf-8"))
        except (OSError, yaml.YAMLError) as exc:
            raise ProfileError(f"cannot read RL profile {path}: {exc}") from exc
        if not isinstance(raw, dict):
            raise ProfileError(f"{path}: expected a YAML mapping")

        allowed = {
            "schema_version",
            "workflow",
            "scene",
            "trainer",
            "algorithm",
            "adapter_module",
            "trainer_config",
            "train_task_id",
            "eval_task_id",
            "task_description",
            "action_dof",
            "policy_action_dof",
            "state_dof",
            "cameras",
            "default_num_envs",
            "default_epochs",
            "simulation",
        }
        unknown = sorted(set(raw) - allowed)
        if unknown:
            raise ProfileError(f"{path}: unknown fields: {', '.join(unknown)}")

        def need(name: str, expected: type) -> Any:
            value = raw.get(name)
            if isinstance(value, bool) and expected in {int, float}:
                raise ProfileError(f"{path}: {name} must be {expected.__name__}")
            if not isinstance(value, expected) or (expected is str and not value.strip()):
                raise ProfileError(f"{path}: {name} must be {expected.__name__}")
            return value

        schema_version = need("schema_version", int)
        if schema_version != 1:
            raise ProfileError(f"{path}: unsupported schema_version {schema_version}; expected 1")
        workflow = need("workflow", str)
        if path.stem != workflow:
            raise ProfileError(f"{path}: filename must match workflow {workflow!r}")

        trainer_config = (path.parent / need("trainer_config", str)).resolve()
        if not trainer_config.is_file():
            raise ProfileError(f"{path}: trainer config does not exist: {trainer_config}")
        action_dof = int(need("action_dof", int))
        policy_action_dof = int(need("policy_action_dof", int))
        state_dof = int(need("state_dof", int))
        if action_dof <= 0 or policy_action_dof <= 0 or state_dof <= 0:
            raise ProfileError(f"{path}: action/state dimensions must be positive")
        if policy_action_dof > action_dof:
            raise ProfileError(f"{path}: policy_action_dof cannot exceed action_dof")
        cameras = need("cameras", list)
        if any(not isinstance(camera, str) or not camera for camera in cameras):
            raise ProfileError(f"{path}: cameras must be a string list")
        if len(cameras) != len(set(cameras)):
            raise ProfileError(f"{path}: cameras must not contain duplicates")
        trainer = need("trainer", str)
        if trainer not in BACKEND_MODULES:
            raise ProfileError(f"{path}: unsupported trainer {trainer!r}")
        adapter_module = raw.get("adapter_module")
        if adapter_module is not None:
            if not isinstance(adapter_module, str) or not adapter_module.strip():
                raise ProfileError(f"{path}: adapter_module must be a non-empty import path")
            if any(not part.isidentifier() for part in adapter_module.split(".")):
                raise ProfileError(f"{path}: invalid adapter_module import path {adapter_module!r}")
        if trainer == "rlinf" and not adapter_module:
            raise ProfileError(f"{path}: RLinf profiles require adapter_module")
        if trainer_config.suffix != ".yaml":
            raise ProfileError(f"{path}: trainer_config must be a .yaml file")
        default_num_envs = int(need("default_num_envs", int))
        default_epochs = int(need("default_epochs", int))
        if default_num_envs <= 0 or default_epochs <= 0:
            raise ProfileError(f"{path}: default_num_envs and default_epochs must be positive")

        simulation_raw = need("simulation", dict)
        simulation_unknown = sorted(set(simulation_raw) - {"env_spacing", "presets", "enable_cameras"})
        if simulation_unknown:
            raise ProfileError(f"{path}: unknown simulation fields: {', '.join(simulation_unknown)}")
        env_spacing = simulation_raw.get("env_spacing")
        if isinstance(env_spacing, bool) or not isinstance(env_spacing, int | float) or env_spacing <= 0:
            raise ProfileError(f"{path}: simulation.env_spacing must be positive")
        presets = simulation_raw.get("presets")
        if not isinstance(presets, str) or not presets.strip():
            raise ProfileError(f"{path}: simulation.presets must be a non-empty string")
        enable_cameras = simulation_raw.get("enable_cameras")
        if not isinstance(enable_cameras, bool):
            raise ProfileError(f"{path}: simulation.enable_cameras must be bool")
        if cameras and not enable_cameras:
            raise ProfileError(f"{path}: camera observations require simulation.enable_cameras=true")
        return cls(
            schema_version=schema_version,
            workflow=workflow,
            scene=need("scene", str),
            trainer=trainer,
            algorithm=need("algorithm", str),
            adapter_module=adapter_module,
            trainer_config=trainer_config,
            train_task_id=need("train_task_id", str),
            eval_task_id=need("eval_task_id", str),
            task_description=need("task_description", str),
            action_dof=action_dof,
            policy_action_dof=policy_action_dof,
            cameras=tuple(cameras),
            state_dof=state_dof,
            default_num_envs=default_num_envs,
            default_epochs=default_epochs,
            simulation=SimulationProfile(
                env_spacing=float(env_spacing),
                presets=presets,
                enable_cameras=enable_cameras,
            ),
        )


def profile_dir() -> Path:
    return Path(__file__).resolve().parents[1] / "profiles"


def available_profiles() -> dict[str, Path]:
    profiles: dict[str, Path] = {}
    for path in sorted(profile_dir().glob("*.yaml")):
        profile = RLProfile.load(path)
        if profile.workflow in profiles:
            raise ProfileError(f"duplicate RL profile for {profile.workflow!r}")
        profiles[profile.workflow] = path
    return profiles


def load_profile(workflow: str) -> RLProfile:
    profiles = available_profiles()
    try:
        return RLProfile.load(profiles[workflow])
    except KeyError as exc:
        known = ", ".join(sorted(profiles)) or "none"
        raise ProfileError(f"workflow {workflow!r} has no RL profile; available: {known}") from exc

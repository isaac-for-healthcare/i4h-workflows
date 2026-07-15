# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from common.config import environment_blocks, environment_config_path


@dataclass(frozen=True)
class ArenaEnvironmentConfig:
    env_id: str
    robot_type: str | None = None
    max_timesteps: int | None = None
    home_joint_pos_rad: list[float] | None = None
    bridge_port: int | None = None
    description: str | None = None


def load_arena_configs() -> dict[str, ArenaEnvironmentConfig]:
    configs = {}
    for env_id, values in environment_blocks().items():
        robot_values = dict(values.get("robot") or {})
        arena_values = dict(values.get("arena") or {})
        arena_values["robot_type"] = robot_values.get("type") or values.get("robot_type")
        arena_values["home_joint_pos_rad"] = robot_values.get("home_joint_pos_rad") or values.get("home_joint_pos_rad")
        configs[env_id] = ArenaEnvironmentConfig(env_id=env_id, **arena_values)
    return configs


def arena_config_path() -> Path:
    return environment_config_path()


def get_arena_config(env_id: str) -> ArenaEnvironmentConfig:
    configs = load_arena_configs()
    return configs.get(env_id, ArenaEnvironmentConfig(env_id=env_id))

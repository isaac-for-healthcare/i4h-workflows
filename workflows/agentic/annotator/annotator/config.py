# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from typing import Any

from common.config import get_environment_config


def environment_block(env_id: str) -> dict[str, Any]:
    return get_environment_config(env_id)


def task_description_for_env(env_id: str, override: str | None = None) -> str:
    if override:
        return override
    env_config = environment_block(env_id)
    policy = env_config.get("policy") or {}
    dataset = env_config.get("dataset") or {}
    return (
        dataset.get("task_description")
        or policy.get("task_description")
        or policy.get("language_instruction")
        or "Perform the recorded manipulation task."
    )


def camera_names_for_env(env_id: str, override: list[str] | None = None) -> list[str]:
    if override:
        return list(override)
    env_config = environment_block(env_id)
    return [str(name) for name in ((env_config.get("zenoh") or {}).get("camera_names") or [])]


def csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]

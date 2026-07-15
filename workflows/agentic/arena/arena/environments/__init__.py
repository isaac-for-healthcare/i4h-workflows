# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false
"""Registered agentic Arena environments.

Environment classes are discovered from ``*_environment.py`` modules and
matched with ``config/environments/<env>.yaml``.
"""

from __future__ import annotations

import importlib
import inspect
import pkgutil

from arena.environments.core.base import AgenticEnvironmentBase
from common.config import environment_blocks, get_env_metadata


def _discover_environment_classes() -> tuple[type[AgenticEnvironmentBase], ...]:
    classes: dict[str, type[AgenticEnvironmentBase]] = {}
    for module_info in pkgutil.iter_modules(__path__, prefix=f"{__name__}."):
        if not module_info.name.endswith("_environment"):
            continue
        module = importlib.import_module(module_info.name)
        for value in vars(module).values():
            if not (
                isinstance(value, type)
                and issubclass(value, AgenticEnvironmentBase)
                and value is not AgenticEnvironmentBase
                and not inspect.isabstract(value)
            ):
                continue
            if value.__module__ != module.__name__:
                continue
            if value.__name__.startswith("_"):
                continue
            env_id = value.name
            if not env_id:
                raise ValueError(f"{value.__module__}.{value.__name__} must define a non-empty name")
            if env_id in classes:
                other = classes[env_id]
                raise ValueError(
                    f"duplicate Arena env name {env_id!r}: "
                    f"{other.__module__}.{other.__name__} and {value.__module__}.{value.__name__}"
                )
            classes[str(env_id)] = value

    configured = set(environment_blocks())
    discovered = set(classes)
    missing_yaml = discovered - configured
    if missing_yaml:
        raise ValueError("Arena env classes missing config/environments YAML: " + ", ".join(sorted(missing_yaml)))
    missing_class = configured - discovered
    if missing_class:
        raise ValueError("config/environments YAML missing Arena env class: " + ", ".join(sorted(missing_class)))

    for env_id, cls in classes.items():
        metadata = get_env_metadata(env_id)
        cls.description = metadata.description
    return tuple(classes[env_id] for env_id in sorted(classes))


_ENV_CLASSES = _discover_environment_classes()

ENVIRONMENTS: dict[str, type[AgenticEnvironmentBase]] = {str(cls.name): cls for cls in _ENV_CLASSES}


def environment_choices() -> tuple[str, ...]:
    return tuple(ENVIRONMENTS.keys())


def get_environment(name: str) -> type[AgenticEnvironmentBase]:
    try:
        return ENVIRONMENTS[name]
    except KeyError as exc:
        choices = ", ".join(environment_choices())
        raise ValueError(f"unknown environment {name!r}; choose one of: {choices}") from exc


__all__ = [
    "AgenticEnvironmentBase",
    "ENVIRONMENTS",
    "environment_choices",
    "get_environment",
]

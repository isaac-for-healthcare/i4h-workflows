# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy backends for the GR00T N1.7 stack."""

from __future__ import annotations

import argparse
import importlib
from collections.abc import Callable
from dataclasses import dataclass

from common.config import policy_routings_for_stack

_STACK = "gr00t_n17"


@dataclass(frozen=True)
class PolicyBackend:
    env_id: str
    description: str
    infer_module: str
    train_module: str | None = None
    add_args_attr: str = "add_args"
    run_attr: str = "run"

    @property
    def train_module_path(self) -> str:
        return self.train_module or f"policy.{self.env_id}.train.train"

    def load_run(self) -> Callable[[argparse.Namespace], None]:
        return getattr(importlib.import_module(self.infer_module), self.run_attr)

    def load_add_args(self) -> Callable[[argparse.ArgumentParser], None]:
        return getattr(importlib.import_module(self.infer_module), self.add_args_attr)


BACKENDS: tuple[PolicyBackend, ...] = tuple(
    PolicyBackend(
        env_id=routing.env_id,
        description=routing.description,
        infer_module=routing.infer_module,
        train_module=routing.train_module,
        add_args_attr=routing.add_args_attr,
        run_attr=routing.run_attr,
    )
    for routing in policy_routings_for_stack(_STACK)
)

_BY_ENV: dict[str, PolicyBackend] = {b.env_id: b for b in BACKENDS}


def get_backend(env_id: str) -> PolicyBackend:
    try:
        return _BY_ENV[env_id]
    except KeyError as exc:
        choices = ", ".join(sorted(_BY_ENV))
        raise KeyError(f"unknown policy env {env_id!r}; choose one of: {choices}") from exc


def known_env_ids() -> tuple[str, ...]:
    return tuple(b.env_id for b in BACKENDS)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic RSL-RL runner configuration loaded from a workflow RL profile."""

from __future__ import annotations

import os
from copy import deepcopy
from dataclasses import field
from pathlib import Path
from typing import Any

import yaml
from isaaclab.utils import configclass
from isaaclab_rl.rsl_rl import RslRlOnPolicyRunnerCfg, RslRlPpoActorCriticCfg, RslRlPpoAlgorithmCfg


def _load() -> dict[str, Any]:
    path_value = os.environ.get("I4H_RL_TRAINER_CONFIG", "").strip()
    if not path_value:
        raise RuntimeError("I4H_RL_TRAINER_CONFIG is required when loading an RSL-RL runner configuration")
    path = Path(path_value)
    try:
        raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise RuntimeError(f"cannot read RSL-RL trainer config {path}: {exc}") from exc
    if not isinstance(raw, dict) or raw.get("schema_version") != 1 or raw.get("backend") != "rsl_rl":
        raise RuntimeError(f"{path}: expected schema_version 1 and backend rsl_rl")
    return raw


_CONFIG = _load()
_RUNNER = _CONFIG["runner"]
_POLICY = _CONFIG["policy"]
_ALGORITHM = _CONFIG["algorithm"]


@configclass
class ProfiledRslRlRunnerCfg(RslRlOnPolicyRunnerCfg):
    """Materialize the selected declarative trainer config for Isaac Lab."""

    num_steps_per_env: int = _RUNNER["num_steps_per_env"]
    max_iterations: int = _RUNNER["max_iterations"]
    save_interval: int = _RUNNER["save_interval"]
    experiment_name: str = _RUNNER["experiment_name"]
    obs_groups = field(default_factory=lambda: deepcopy(_RUNNER["obs_groups"]))
    policy: RslRlPpoActorCriticCfg = RslRlPpoActorCriticCfg(**_POLICY)
    algorithm: RslRlPpoAlgorithmCfg = RslRlPpoAlgorithmCfg(**_ALGORITHM)

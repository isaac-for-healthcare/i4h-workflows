# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy trainer-backend discovery for the lightweight RL CLI."""

from __future__ import annotations

import importlib
from types import ModuleType

BACKEND_MODULES = {
    "rlinf": "i4h_rl.backends.rlinf",
    "rsl_rl": "i4h_rl.backends.rsl_rl",
}


def load_backend(trainer: str) -> ModuleType:
    try:
        module_name = BACKEND_MODULES[trainer]
    except KeyError as exc:
        raise RuntimeError(f"unsupported RL trainer backend: {trainer}") from exc
    return importlib.import_module(module_name)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Lazy workflow-adapter loading for RL backends."""

from __future__ import annotations

import importlib
from types import ModuleType


def load_adapter(module_name: str | None, *, required: bool = False) -> ModuleType | None:
    if not module_name:
        if required:
            raise RuntimeError("the selected RL profile does not declare adapter_module")
        return None
    try:
        return importlib.import_module(module_name)
    except ImportError as exc:
        raise RuntimeError(f"cannot import RL adapter {module_name!r}: {exc}") from exc

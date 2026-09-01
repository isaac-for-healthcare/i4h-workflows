# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic RLinf extension that delegates to the selected workflow adapter."""

from __future__ import annotations

import os

from i4h_rl.adapter_loader import load_adapter

_ADAPTER_ENV = "I4H_RL_ADAPTER_MODULE"
_registered: set[str] = set()


def register() -> None:
    """Load and register the adapter declared by the selected RL profile."""
    module_name = os.environ.get(_ADAPTER_ENV)
    if not module_name:
        raise RuntimeError(f"{_ADAPTER_ENV} is not set")
    if module_name in _registered:
        return
    adapter = load_adapter(module_name, required=True)
    adapter_register = getattr(adapter, "register", None)
    if not callable(adapter_register):
        raise RuntimeError(f"RLinf adapter {module_name!r} does not define register()")
    adapter_register()
    _registered.add(module_name)

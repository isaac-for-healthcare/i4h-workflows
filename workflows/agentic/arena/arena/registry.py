# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Backwards-compatible shim.

The real registry now lives in :mod:`arena.environments`. Importers that
expect ``ENVIRONMENTS``, ``environment_choices`` or ``get_environment`` from
``arena.registry`` get re-exports here.
"""

from arena.environments import ENVIRONMENTS, AgenticEnvironmentBase, environment_choices, get_environment

__all__ = [
    "AgenticEnvironmentBase",
    "ENVIRONMENTS",
    "environment_choices",
    "get_environment",
]

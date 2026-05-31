# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Policy backends for the GR00T N1.5 stack."""

from __future__ import annotations

from common.policy_stack import PolicyBackend, build_backends
from common.policy_stack import get_backend as _get_backend
from common.policy_stack import known_env_ids as _known_env_ids

_STACK = "gr00t_n15"

BACKENDS: tuple[PolicyBackend, ...] = build_backends(_STACK)


def get_backend(env_id: str) -> PolicyBackend:
    return _get_backend(BACKENDS, env_id)


def known_env_ids() -> tuple[str, ...]:
    return _known_env_ids(BACKENDS)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Registry-driven policy CLI for the GR00T N1.5 stack."""

from __future__ import annotations

from common.policy_stack import policy_cli_main
from policy.registry import BACKENDS


def main() -> None:
    policy_cli_main(BACKENDS)


if __name__ == "__main__":
    main()

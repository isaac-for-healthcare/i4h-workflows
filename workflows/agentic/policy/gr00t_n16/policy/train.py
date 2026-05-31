# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Top-level train CLI dispatcher for gr00t_n16 (GR00T N1.6 stack).

Mirrors :mod:`policy.cli` for inference: picks ``--env`` and forwards the
remaining argv to ``policy.<env_id>.train:main``.
"""

from __future__ import annotations

import argparse
import importlib
import os
import sys

from policy.registry import get_backend, known_env_ids


def main() -> None:
    base = argparse.ArgumentParser(add_help=False)
    base.add_argument("--env", choices=known_env_ids(), required=True)
    known, rest = base.parse_known_args()

    os.environ["AGENTIC_POLICY_ENV_ID"] = known.env
    module = importlib.import_module(get_backend(known.env).train_module_path)
    sys.argv = [sys.argv[0], *rest]
    module.main()


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-launch shim for Isaac-GR00T's ``launch_finetune.py``.

Currently a thin ``runpy`` wrapper — no gr00t patches. Kept as a hook for
future env / argv preparation that needs to land before the launcher's
``tyro.cli`` call.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m policy.<env>.train._launcher <launch_finetune.py> [args...]")
    launch_script = sys.argv[1]
    sys.argv = [launch_script, *sys.argv[2:]]
    runpy.run_path(launch_script, run_name="__main__")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pre-launch shim for Isaac-GR00T's ``launch_finetune.py``.

Currently a thin ``runpy`` wrapper — no gr00t patches. Kept as a hook for
future env / argv preparation that needs to land before the launcher's
``tyro.cli`` call.

Run with ``-m`` rather than invoked as a path: that puts the working directory
the fine-tune subprocess is given, the Isaac-GR00T checkout root, at the front
of ``sys.path``, which is where the launcher resolves its own ``gr00t`` imports
from. Running the launch script directly would put the script's own directory
there instead.
"""

from __future__ import annotations

import runpy
import sys


def main() -> None:
    if len(sys.argv) < 2:
        sys.exit("usage: python -m i4h_tasks.gr00t_n17._launcher <launch_finetune.py> [args...]")
    launch_script = sys.argv[1]
    sys.argv = [launch_script, *sys.argv[2:]]
    runpy.run_path(launch_script, run_name="__main__")


if __name__ == "__main__":
    main()

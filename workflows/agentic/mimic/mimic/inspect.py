# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse

from common.utils import resolve_path
from mimic.hdf5 import inspect_file


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an Agentic HDF5 demo file.")
    parser.add_argument("path", help="HDF5 file to inspect.")
    parser.add_argument("--env", required=True, help="Environment id for resolving relative paths under runs/<env>.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    path = resolve_path(args.path, args.env)
    if not path.exists():
        raise FileNotFoundError(path)

    summaries = inspect_file(path)
    print(f"{path}: {len(summaries)} demos")
    for summary in summaries:
        cameras = ", ".join(f"{name}={shape}" for name, shape in summary.cameras.items()) or "none"
        print(
            f"- {summary.name}: length={summary.length}, actions={summary.action_shape}, "
            f"state={summary.state_shape}, cameras={cameras}, success={summary.success}"
        )


if __name__ == "__main__":
    main()

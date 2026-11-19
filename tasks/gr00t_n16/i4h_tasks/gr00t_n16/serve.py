# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve an i4h GR00T N1.6 checkpoint with GR00T 1.6's own inference server.

The task manifest pins the checkpoint's repository and revision. This module downloads
that revision, unless a local checkpoint is given, and runs GR00T's
``gr00t/eval/run_gr00t_server.py``, which Arena's GR00T client drives::

    tasks/gr00t_n16/.venv/bin/python -m i4h_tasks.gr00t_n16.serve locomanip_push_cart --port 5555
"""

from __future__ import annotations

import argparse
import logging
import os

logger = logging.getLogger("i4h_tasks.gr00t_n16.serve")

#: Tasks served here; each one's checkpoint is in manifest/<task>.yaml.
TASKS = ("locomanip_push_cart", "locomanip_tray_pick_and_place")


def resolve_model_path(path: str | None, repo: str | None, revision: str | None) -> str:
    """A local checkpoint as given, or the manifest's pinned revision from the hub."""
    if path:
        expanded = os.path.expanduser(path)
        if not os.path.exists(expanded):
            raise FileNotFoundError(f"no checkpoint at {expanded}")
        return expanded
    if not repo:
        raise ValueError("the task names no published checkpoint; pass --model-path")
    from huggingface_hub import snapshot_download  # noqa: PLC0415

    logger.info("downloading %s@%s", repo, revision or "main")
    return snapshot_download(repo_id=repo, revision=revision)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="i4h_tasks.gr00t_n16.serve", description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--model-path", help="local or finetuned checkpoint; default: the manifest's pinned repo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from gr00t.data.embodiment_tags import EmbodimentTag  # noqa: PLC0415
    from gr00t.eval.run_gr00t_server import ServerConfig  # noqa: PLC0415
    from gr00t.eval.run_gr00t_server import main as run_server  # noqa: PLC0415

    from i4h_common.training import task_spec  # noqa: PLC0415

    model = task_spec(f"gr00t_n16/{args.task}").model
    path = resolve_model_path(args.model_path, model.get("repo"), model.get("revision"))
    tag = str(model.get("embodiment_tag", "new_embodiment"))
    run_server(
        ServerConfig(
            model_path=path,
            embodiment_tag=EmbodimentTag(tag),
            device=args.device,
            host=args.host,
            port=args.port,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve an i4h GR00T N1.7 checkpoint with GR00T 1.7's own inference server.

GR00T's ``gr00t/eval/run_gr00t_server.py`` loads a checkpoint whose embodiment
modality config is already registered. The i4h checkpoints use modality configs
registered in this package (``config.py``, ``config_catheter.py``), and their vision
backbone needs the patch-embedding fix in ``_patch_embed``. This module applies both
and then runs GR00T's server, which speaks the protocol of GR00T's ``PolicyClient``::

    tasks/gr00t_n17/.venv/bin/python -m i4h_tasks.gr00t_n17.serve scissor_pick_and_place --port 5555
"""

from __future__ import annotations

import argparse
import importlib
import logging
import os

logger = logging.getLogger("i4h_tasks.gr00t_n17.serve")

#: Tasks served here; each one's checkpoint and modality config are in manifest/<task>.yaml.
TASKS = ("scissor_pick_and_place", "catheter_navigation")

#: Registered when a task names none: the SO-ARM's.
DEFAULT_MODALITY_CONFIG = "config"


def register_modalities(module: str) -> None:
    """Import a sibling modality config for its registration side effect.

    Registration has to happen before the policy reads the checkpoint processor's
    embodiment metadata. Names are checked against the modules that exist rather than
    passed to ``import_module`` as given, so a typo in a manifest names the
    alternatives instead of raising a bare ImportError.
    """
    here = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(here, f"{module}.py")):
        available = sorted(name[:-3] for name in os.listdir(here) if name.startswith("config") and name.endswith(".py"))
        raise ValueError(f"no modality config {module!r} in this stack; have {available}")
    importlib.import_module(f"{__package__}.{module}")


def resolve_model_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    # A filesystem path that is not there is a missing checkpoint, not a hub repo.
    # Saying so beats the HFValidationError that asking the hub about an absolute path
    # produces, which reads as a credentials or naming problem when the real answer is
    # that training never got far enough to save. A repo id is `namespace/name`, so one
    # separator is still hub-shaped.
    if os.path.isabs(expanded) or expanded.startswith(("./", "../", "~")) or expanded.count("/") > 1:
        raise FileNotFoundError(f"no checkpoint at {expanded}")
    from huggingface_hub import snapshot_download

    logger.info("downloading %s", path)
    return snapshot_download(repo_id=path)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="i4h_tasks.gr00t_n17.serve", description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--model-path", help="local or finetuned checkpoint; default: the manifest's repo")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from gr00t.eval.run_gr00t_server import ServerConfig
    from gr00t.eval.run_gr00t_server import main as run_server

    from i4h_common.training import task_spec
    from i4h_tasks.gr00t_n17._patch_embed import install_linear_patch_embed

    model = task_spec(f"gr00t_n17/{args.task}").model
    path = args.model_path or model.get("repo")
    if not path:
        parser.error(f"{args.task} names no published checkpoint; pass --model-path")
    install_linear_patch_embed()
    register_modalities(str(model.get("modality_config") or DEFAULT_MODALITY_CONFIG))
    run_server(
        ServerConfig(
            model_path=resolve_model_path(str(path)),
            embodiment_tag=str(model.get("embodiment_tag", "new_embodiment")),
            device=args.device,
            host=args.host,
            port=args.port,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

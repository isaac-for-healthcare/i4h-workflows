# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve the liver-scan PI0 checkpoint with openpi's own websocket policy server.

openpi's ``scripts/serve_policy.py`` serves the training configs registered in openpi
itself; the liver-scan config is registered in this package, and a locally trained
checkpoint carries its own normalization statistics. This module loads the policy the
way ``serve_policy.py`` does, compiles it with one warm-up inference, and serves it on
openpi's protocol, which Arena's openpi client speaks::

    tasks/openpi_pi0/.venv/bin/python -m i4h_tasks.openpi_pi0.serve ultrasound_liver_scan --port 8000
"""

from __future__ import annotations

import argparse
import dataclasses
import logging
import os
import time
from pathlib import Path
from typing import Any

import numpy as np

from i4h_tasks.openpi_pi0.runtime import configure_xla, cuda_compute_capability

logger = logging.getLogger("i4h_tasks.openpi_pi0.serve")

#: Tasks served here; each one's checkpoint is pinned in manifest/<task>.yaml.
TASKS = ("ultrasound_liver_scan",)
#: Image size PI0 was trained on; the client pads every camera to it.
IMAGE_SIZE = (224, 224)
#: Franka joints in the state vector.
STATE_DIM = 7


def checkpoint_asset_id(checkpoint: str) -> str:
    """Resolve the single normalization identity saved by OpenPI, without a dataset."""
    assets = Path(checkpoint).expanduser().resolve() / "assets"
    stats = sorted(path for path in assets.rglob("norm_stats.json") if path.is_file())
    if len(stats) != 1 or stats[0].parent == assets:
        found = [str(path.relative_to(assets)) for path in stats]
        raise ValueError(
            f"{checkpoint}: expected exactly one normalization asset under assets/<asset_id>/norm_stats.json; "
            f"found {len(stats)}: {found}"
        )
    return stats[0].parent.relative_to(assets).as_posix()


def load_policy(model: dict[str, Any], checkpoint: str | None = None) -> Any:
    """The manifest's checkpoint, or a locally trained ``checkpoint`` with its own statistics."""
    asset_id = checkpoint_asset_id(checkpoint) if checkpoint else None
    configure_xla(cuda_compute_capability())
    from openpi.policies import policy_config

    from i4h_tasks.openpi_pi0.ultrasound.config import get_config

    path = checkpoint or str(model.get("repo", ""))
    if not path:
        raise ValueError("no model repo in the manifest and no --checkpoint")
    if not checkpoint and not Path(path).expanduser().exists():
        from huggingface_hub import snapshot_download

        cache_dir = os.environ.get("ULTRASOUND_MODEL_CACHE", os.path.expanduser("~/.cache/ultrasound_models"))
        path = snapshot_download(repo_id=path, cache_dir=cache_dir)
    config_name = str(model.get("config", "robotic_ultrasound"))
    repo_id = str(model.get("repo_id", "i4h/sim_liver_scan"))
    config = get_config(name=config_name, repo_id=repo_id)
    if asset_id is not None:
        from openpi.shared import normalize

        assets_dir = Path(path).expanduser().resolve() / "assets"
        # Validate the saved statistics before create_trained_policy loads weights.
        normalize.load(assets_dir / asset_id)
        config = dataclasses.replace(
            config,
            data=dataclasses.replace(
                config.data,
                assets=dataclasses.replace(config.data.assets, assets_dir=str(assets_dir), asset_id=asset_id),
            ),
        )
    logger.info("loading PI0 %s (config=%s repo_id=%s)", path, config_name, repo_id)
    return policy_config.create_trained_policy(config, path)


def warm_up(policy: Any, prompt: str) -> None:
    """Run one inference on a blank request so JAX compiles PI0 before the first client connects.

    JAX compiles the sampling loop on its first call; left to the first real request,
    that compile ran inside the client's first step and timed it out (NVBug 6935977).
    The request has the client's keys, shapes and dtypes, so nothing compiles again.
    """
    blank = np.zeros((*IMAGE_SIZE, 3), dtype=np.uint8)
    started = time.monotonic()
    policy.infer(
        {
            "observation/state": np.zeros(STATE_DIM, dtype=np.float64),
            "observation/image": blank,
            "observation/wrist_image": blank,
            "prompt": prompt,
        }
    )
    logger.info("compiled PI0 with a warm-up inference in %.1fs", time.monotonic() - started)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="i4h_tasks.openpi_pi0.serve", description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--checkpoint", help="locally trained checkpoint; default: the manifest's model")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO, force=True)

    from i4h_common.training import task_spec

    spec = task_spec(f"openpi_pi0/{args.task}")
    policy = load_policy(spec.model, args.checkpoint)
    warm_up(policy, spec.prompt or spec.summary)

    from openpi.serving.websocket_policy_server import WebsocketPolicyServer

    WebsocketPolicyServer(policy=policy, host=args.host, port=args.port, metadata=policy.metadata).serve_forever()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

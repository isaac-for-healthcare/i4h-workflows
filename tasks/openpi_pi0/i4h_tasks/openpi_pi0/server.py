# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""openpi PI0 backend (Franka ultrasound liver scan).

Same protocol as the GR00T stacks — that is the point of putting it in
:class:`i4h_common.server.PolicyServer`. PI0 differs in two ways that matter here:
it wants a flat ``state`` vector rather than named modalities, and it returns a
long horizon (50) that the arena side consumes over many ticks.
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path
from typing import Any

import numpy as np

from i4h_common.bus.messages import ObsFrame
from i4h_common.server import ActionContract, PolicyServer, Session

logger = logging.getLogger("i4h_tasks.openpi_pi0")


class OpenPiPi0Server(PolicyServer):
    """Serve ``openpi_pi0/*`` tasks."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._policies: dict[str, Any] = {}

    def load(self, session: Session) -> None:
        key = self._cache_key(session)
        if key in self._policies:
            return

        from openpi.policies import policy_config

        model = session.model
        path = session.checkpoint or str(model.get("repo", ""))
        if not path:
            raise ValueError(f"{session.task_id}: no model repo in the manifest and no --checkpoint")
        if not session.checkpoint and not Path(path).expanduser().exists():
            from huggingface_hub import snapshot_download

            cache_dir = os.environ.get(
                "ULTRASOUND_MODEL_CACHE",
                os.path.expanduser("~/.cache/ultrasound_models"),
            )
            path = snapshot_download(repo_id=path, cache_dir=cache_dir)
        from i4h_tasks.openpi_pi0.ultrasound.config import get_config

        config_name = str(model.get("config", "robotic_ultrasound"))
        repo_id = str(model.get("repo_id", "i4h/sim_liver_scan"))
        logger.info("loading PI0 %s (config=%s repo_id=%s)", path, config_name, repo_id)
        self._policies[key] = policy_config.create_trained_policy(
            get_config(name=config_name, repo_id=repo_id),
            path,
        )

    def _cache_key(self, session: Session) -> str:
        return f"{session.checkpoint or session.model.get('repo', '')}|{session.model.get('config', '')}"

    def unload(self, session: Session) -> None:
        pass  # cached across episodes

    def action_contract(self, session: Session) -> ActionContract:
        return ActionContract(space="ee_pose", layout="delta_axis_angle", dof=6)

    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        policy = self._policies.get(self._cache_key(session))
        if policy is None:
            return None

        from openpi_client import image_tools

        images = session.images(frame)
        observation: dict[str, Any] = {
            "observation/state": np.asarray(frame.state, dtype=np.float64)[:7],
            "prompt": session.prompt,
        }
        if "room" in images:
            observation["observation/image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images["room"], 224, 224)
            )
        if "wrist" in images:
            observation["observation/wrist_image"] = image_tools.convert_to_uint8(
                image_tools.resize_with_pad(images["wrist"], 224, 224)
            )

        result = policy.infer(observation)
        actions = np.atleast_2d(np.asarray(result["actions"], dtype=np.float32))
        # The simulator's relative IK action is exactly dx,dy,dz,axis-angle.
        return actions[: min(5, session.execution_steps, len(actions)), :6]


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="openpi-pi0-server")
    parser.add_argument("--namespace", required=True)
    parser.add_argument("--verbose", action="store_true")
    parser.add_argument(
        "--preload",
        action="append",
        default=[],
        help="load this task's checkpoint at startup instead of on the first spec",
    )
    parser.add_argument("--checkpoint", default="", help="checkpoint override used while preloading")
    parser.add_argument("--preload-only", action="store_true", help="load requested checkpoints and exit")
    args = parser.parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="[openpi-pi0] %(message)s",
    )
    server = OpenPiPi0Server(namespace=args.namespace)
    if args.preload_only:
        if not args.preload:
            parser.error("--preload-only requires --preload")
        server.preload_only(tuple(args.preload), checkpoint=args.checkpoint)
        return 0
    server.serve_forever(
        preload=tuple(args.preload),
        preload_checkpoint=args.checkpoint,
    )
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

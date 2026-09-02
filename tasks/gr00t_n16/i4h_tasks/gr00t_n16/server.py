# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N16 backend.

Same protocol as every other stack — see :class:`i4h_common.server.PolicyServer`.
Separate project purely because its dependency pins differ; the code that
differs from N1.5 is the modality split and the default data_config, which is
why this file is short.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import Any

import numpy as np

from i4h_common.bus.messages import ObsFrame
from i4h_common.paths import workflow_root
from i4h_common.server import ActionContract, PolicyServer, Session

logger = logging.getLogger("i4h_tasks.gr00t_n16")


#: G1 whole-body control emits one flat action vector, not per-limb modalities.
MODALITY_KEYS = ("whole_body",)


class Gr00tN16Server(PolicyServer):
    """Serves ``gr00t_n16/*`` tasks (Unitree G1 loco-manipulation)."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._policies: dict[str, Any] = {}

    def load(self, session: Session) -> None:
        key = self._cache_key(session)
        if key in self._policies:
            self._policies[key].reset()
            return
        model = session.model
        repo = str(model.get("repo", ""))
        if not session.checkpoint and not repo:
            raise ValueError(f"{session.task_id}: no model repo and no --checkpoint")

        # N1.6 G1 checkpoints require the Arena joint remapper and WBC command
        # tail, rather than the N1.5 DATA_CONFIG_MAP interface.
        from i4h_tasks.gr00t_n16._finetune import _load_modality_config
        from i4h_tasks.gr00t_n16.locomanip.infer.closedloop_policy import G1LocomanipClosedloopPolicy

        train = dict(self._declaration(session.task_id).get("train", {}) or {})
        modality_config = train.get("modality_config_path")
        if modality_config:
            modality_config_path = Path(str(modality_config)).expanduser()
            if not modality_config_path.is_absolute():
                modality_config_path = workflow_root() / modality_config_path
        else:
            modality_config_path = Path(__file__).resolve().parent / "locomanip" / "config_dualcam.py"
        _load_modality_config(modality_config_path.resolve())

        config = dict(model)
        config["language_instruction"] = session.prompt
        logger.info("loading %s with the G1 closed-loop/WBC adapter", session.checkpoint or repo)
        self._policies[key] = G1LocomanipClosedloopPolicy(
            num_envs=1,
            device="cuda",
            model_path_override=session.checkpoint or None,
            model_repo=repo or None,
            model_revision=str(model.get("revision") or "") or None,
            policy_config_data=config,
            config_base_dir=workflow_root(),
        )

    def _cache_key(self, session: Session) -> str:
        return (
            f"{session.task_id}|{session.checkpoint or session.model.get('repo', '')}|"
            f"{session.model.get('data_config', '')}|{session.prompt}"
        )

    def unload(self, session: Session) -> None:
        pass  # keep cached across episodes; reloading dominates wall clock

    def action_contract(self, session: Session) -> ActionContract:
        return ActionContract(space="joint_position", layout="joints", dof=50)

    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        policy = self._policies.get(self._cache_key(session))
        if policy is None:
            return None
        import torch

        images = session.images(frame)
        observation = {
            "policy": {
                "robot_joint_pos": torch.as_tensor(
                    np.asarray(frame.state, dtype=np.float32)[np.newaxis, :],
                    device="cuda",
                )
            },
            "camera_obs": {
                name: torch.as_tensor(pixels[np.newaxis, ...].copy(), device="cuda") for name, pixels in images.items()
            },
        }
        # PolicyServer's wire contract accepts a complete action chunk. The
        # closed-loop adapter's get_action() API is for an in-process simulator
        # loop and returns only one row from its internally buffered chunk.
        # Publishing the native chunk avoids another image/Zenoh round trip for
        # each of the remaining rows.
        action_chunk = policy.get_action_chunk(observation)
        return action_chunk[0].detach().cpu().numpy().astype(np.float32)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gr00t-n16-server")
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
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[gr00t-n16] %(message)s")
    server = Gr00tN16Server(namespace=args.namespace)
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

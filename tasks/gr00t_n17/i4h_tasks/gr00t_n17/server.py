# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N17 backend.

Same protocol as every other stack — see :class:`i4h_common.server.PolicyServer`.
Separate project purely because its dependency pins differ; the code that
differs from N1.5 is the modality split and the default data_config, which is
why this file is short.
"""

from __future__ import annotations

import argparse
import logging
import os
from typing import Any

import numpy as np

from i4h_common.bus.messages import ObsFrame
from i4h_common.config import RobotConfig, get_robot_config
from i4h_common.joint_utils import isaaclab_rad_to_lerobot, lerobot_to_isaaclab_rad
from i4h_common.server import ActionContract, PolicyServer, Session

logger = logging.getLogger("i4h_tasks.gr00t_n17")


MODALITY_KEYS = ("single_arm", "gripper")


class Gr00tN17Server(PolicyServer):
    """Serve ``gr00t_n17/*`` tasks as a drop-in alternative to the N1.5 backend."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._policies: dict[str, Any] = {}

    def load(self, session: Session) -> None:
        key = self._cache_key(session)
        if key in self._policies:
            return
        # Registration is a side effect required before the policy reads the
        # checkpoint processor's embodiment metadata.
        from gr00t.data.embodiment_tags import EmbodimentTag
        from gr00t.policy.gr00t_policy import Gr00tPolicy

        import i4h_tasks.gr00t_n17.config  # noqa: F401

        model = session.model
        path = session.checkpoint or str(model.get("repo", ""))
        if not path:
            raise ValueError(f"{session.task_id}: no model repo and no --checkpoint")
        path = _resolve_model_path(path)
        logger.info("loading %s", path)
        policy = Gr00tPolicy(
            model_path=path,
            embodiment_tag=EmbodimentTag.resolve(str(model.get("embodiment_tag", "new_embodiment"))),
            device="cuda",
        )
        self._policies[key] = policy

    def _cache_key(self, session: Session) -> str:
        return f"{session.checkpoint or session.model.get('repo', '')}|{session.model.get('data_config', '')}"

    def unload(self, session: Session) -> None:
        pass

    def action_contract(self, session: Session) -> ActionContract:
        robot = self._robot(session)
        return ActionContract(
            space="joint_position",
            layout="joints",
            dof=len(robot.joint_names),
            gripper="last",
        )

    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        policy = self._policies.get(self._cache_key(session))
        if policy is None:
            return None
        images = session.images(frame)
        robot = self._robot(session)
        state = self._ordered_state(frame, robot)
        state = isaaclab_rad_to_lerobot(
            state,
            robot.isaaclab_joint_pos_limit_range,
            robot.lerobot_joint_pos_limit_range,
        ).astype(np.float32)
        width = state.shape[-1]
        observation = {
            "video": {name: pixels[np.newaxis, np.newaxis, ...] for name, pixels in images.items()},
            "state": {
                "single_arm": state[np.newaxis, np.newaxis, : width - 1],
                "gripper": state[np.newaxis, np.newaxis, width - 1 :],
            },
            "language": {
                policy.language_key: [[session.prompt]],
            },
        }
        chunk, _info = policy.get_action(observation)
        actions = self._flatten(chunk)
        actions = lerobot_to_isaaclab_rad(
            actions,
            robot.lerobot_joint_pos_limit_range,
            robot.isaaclab_joint_pos_limit_range,
        ).astype(np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("GR00T N1.7 produced non-finite joint actions")
        return actions[: min(session.execution_steps, len(actions))]

    @staticmethod
    def _flatten(chunk: dict[str, Any] | np.ndarray) -> np.ndarray:
        if isinstance(chunk, np.ndarray):
            return np.atleast_2d(chunk).astype(np.float32)
        parts = [_as_steps(_action_value(chunk, key)) for key in MODALITY_KEYS]
        return np.concatenate(parts, axis=-1)

    @staticmethod
    def _robot(session: Session) -> RobotConfig:
        if not session.embodiment:
            raise ValueError(f"{session.task_id}: task declaration has no embodiment")
        robot = get_robot_config(session.embodiment)
        if not robot.isaaclab_joint_pos_limit_range or not robot.lerobot_joint_pos_limit_range:
            raise ValueError(f"{session.task_id}: robot {robot.name!r} has no joint calibration ranges")
        return robot

    @staticmethod
    def _ordered_state(frame: ObsFrame, robot: RobotConfig) -> np.ndarray:
        state = np.asarray(frame.state, dtype=np.float32)
        if not frame.state_names:
            return state
        by_name = {name: index for index, name in enumerate(frame.state_names)}
        missing = [name for name in robot.joint_names if name not in by_name]
        if missing:
            raise ValueError(f"observation is missing robot joints {missing}; got {frame.state_names}")
        return state[[by_name[name] for name in robot.joint_names]]


def _resolve_model_path(path: str) -> str:
    expanded = os.path.expanduser(path)
    if os.path.exists(expanded):
        return expanded
    from huggingface_hub import snapshot_download

    logger.info("downloading %s", path)
    return snapshot_download(repo_id=path)


def _action_value(chunk: dict[str, Any], key: str) -> Any:
    for candidate in (key, f"action.{key}", f"state.{key}"):
        if candidate in chunk:
            return chunk[candidate]
    raise KeyError(f"no action for {key!r}; got {sorted(chunk)}")


def _as_steps(values: Any) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        return array.reshape(-1, 1)
    return np.atleast_2d(array)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gr00t-n17-server")
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
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[gr00t-n17] %(message)s")
    server = Gr00tN17Server(namespace=args.namespace)
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

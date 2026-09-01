# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.5 backend.

The wire protocol lives once in :class:`i4h_common.server.PolicyServer`, so this
module only owns model loading and observation shaping.

Launched by ``run.sh`` when a workflow contains a ``gr00t_n15/*`` node; the arena
process never imports this module, and could not (conflicting torch pins).
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np

from i4h_common.bus.messages import ObsFrame
from i4h_common.config import RobotConfig, get_robot_config
from i4h_common.joint_utils import isaaclab_rad_to_lerobot, lerobot_to_isaaclab_rad
from i4h_common.server import ActionContract, PolicyServer, Session

logger = logging.getLogger("i4h_tasks.gr00t_n15")


#: Modality split used by every N1.5 checkpoint here. The dataset half of the
#: same split lives in the embodiment manifest as action_split/state_split.
MODALITY_KEYS = ("single_arm", "gripper")


class Gr00tN15Server(PolicyServer):
    """Serve any ``gr00t_n15/*`` task declared in this project's manifests."""

    def __init__(self, **kwargs: Any) -> None:
        super().__init__(**kwargs)
        self._policies: dict[str, Any] = {}

    # -- model -----------------------------------------------------------
    def load(self, session: Session) -> None:
        key = self._cache_key(session)
        if key in self._policies:
            logger.info("reusing loaded policy %s", key)
            return

        if session.task_id.endswith("/assemble_trocar"):
            from i4h_tasks.gr00t_n15.assemble_trocar.infer.policy import AssembleTrocarPolicy

            model = session.model
            logger.info("loading Assemble Trocar adapter for %s", session.checkpoint or model.get("repo", ""))
            self._policies[key] = AssembleTrocarPolicy(
                model_path=session.checkpoint or None,
                model_repo=str(model.get("repo") or "") or None,
                model_revision=str(model.get("revision") or "") or None,
                task_description=session.prompt,
                device="cuda",
                action_head_future_tokens=int(model.get("action_head_future_tokens", 32)),
                trt_engine_path=str(model.get("trt_engine_path") or "") or None,
            )
            return

        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        model = session.model
        data_config_name = str(model.get("data_config", "so100_dualcam"))
        try:
            data_config = DATA_CONFIG_MAP[data_config_name]
        except KeyError as exc:
            raise KeyError(f"unknown GR00T data_config {data_config_name!r}; known: {sorted(DATA_CONFIG_MAP)}") from exc

        # --checkpoint overrides the manifest, so a freshly-trained run can be
        # evaluated without editing its task manifest.
        path = session.checkpoint or str(model.get("repo", ""))
        if not path:
            raise ValueError(f"{session.task_id}: no model repo in the manifest and no --checkpoint")

        # The stock data_config names the cameras it was written for
        # (so100_dualcam says video.front), but a checkpoint is trained against
        # whatever the scene actually published. Take the names from the
        # session so the two agree, or loading fails on dataset metadata.
        cameras = list(session.observation.get("cameras") or ())
        if cameras:
            data_config.video_keys = [f"video.{name}" for name in cameras]
            logger.info("video keys: %s", data_config.video_keys)

        logger.info("loading %s (data_config=%s)", path, data_config_name)
        self._policies[key] = Gr00tPolicy(
            model_path=path,
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=str(model.get("embodiment_tag", "new_embodiment")),
            denoising_steps=int(model.get("denoising_steps", 4)),
            # Without this the head builds `future_tokens` from scratch and the
            # checkpoint's weights are silently replaced by random ones, which
            # comes out as NaN on the first inference.
            action_head_future_tokens=int(model.get("action_head_future_tokens", 0)),
            device="cuda",
        )

    def _cache_key(self, session: Session) -> str:
        return (
            f"{session.checkpoint or session.model.get('repo', '')}|"
            f"{session.model.get('data_config', '')}|{session.prompt}"
        )

    def unload(self, session: Session) -> None:
        # Deliberately keep the policy cached: episodes re-spec the same node,
        # and reloading a 3B checkpoint between episodes would dominate wall clock.
        pass

    def action_contract(self, session: Session) -> ActionContract:
        if session.task_id.endswith("/assemble_trocar"):
            return ActionContract(space="joint_position", layout="joints", dof=43)
        robot = self._robot(session)
        return ActionContract(
            space="joint_position",
            layout="joints",
            dof=len(robot.joint_names),
            gripper="last" if robot.action_split and robot.action_split[-1][0] == "gripper" else "none",
        )

    # -- inference -------------------------------------------------------
    def infer(self, session: Session, frame: ObsFrame) -> np.ndarray | None:
        policy = self._policies.get(self._cache_key(session))
        if policy is None:
            return None

        if session.task_id.endswith("/assemble_trocar"):
            images = session.images(frame)
            actions = policy.get_action(
                {
                    "frames": images,
                    "joint_positions": np.asarray(frame.state, dtype=np.float32),
                }
            )
            actions = np.atleast_2d(np.asarray(actions, dtype=np.float32))
            if not np.isfinite(actions).all():
                raise ValueError("Assemble Trocar produced non-finite joint actions")
            return actions[: min(session.execution_steps, len(actions))]

        observation = self._observation(session, frame)
        chunk = policy.get_action(observation)
        actions = self._flatten(chunk)
        robot = self._robot(session)
        actions = lerobot_to_isaaclab_rad(
            actions,
            robot.lerobot_joint_pos_limit_range,
            robot.isaaclab_joint_pos_limit_range,
        ).astype(np.float32)
        if not np.isfinite(actions).all():
            raise ValueError("GR00T produced non-finite joint actions")
        horizon = min(session.execution_steps, len(actions))
        return actions[:horizon]

    def _observation(self, session: Session, frame: ObsFrame) -> dict[str, Any]:
        """Shape the frame the way this checkpoint's data_config expects."""
        images = session.images(frame)
        observation: dict[str, Any] = {}
        for name, pixels in images.items():
            observation[f"video.{name}"] = pixels[np.newaxis, ...]

        # float64 and a bare string, matching the workflow runtime — the modality
        # transform is sensitive to both.
        robot = self._robot(session)
        state = self._ordered_state(frame, robot)
        state = isaaclab_rad_to_lerobot(
            state,
            robot.isaaclab_joint_pos_limit_range,
            robot.lerobot_joint_pos_limit_range,
        ).astype(np.float64)
        for group, (start, stop) in self._state_slices(session, state.shape[-1]).items():
            observation[f"state.{group}"] = state[np.newaxis, start:stop]

        # The key the data_config declares in language_keys. Getting it wrong
        # means the instruction never reaches the model.
        observation["annotation.human.task_description"] = session.prompt
        return observation

    @staticmethod
    def _robot(session: Session) -> RobotConfig:
        if not session.embodiment:
            raise ValueError(f"{session.task_id}: task declaration has no embodiment")
        robot = get_robot_config(session.embodiment)
        if not robot.isaaclab_joint_pos_limit_range or not robot.lerobot_joint_pos_limit_range:
            raise ValueError(
                f"{session.task_id}: robot {robot.name!r} has no IsaacLab/LeRobot joint calibration ranges"
            )
        return robot

    @staticmethod
    def _ordered_state(frame: ObsFrame, robot: RobotConfig) -> np.ndarray:
        state = np.asarray(frame.state, dtype=np.float64)
        if not frame.state_names:
            return state
        by_name = {name: index for index, name in enumerate(frame.state_names)}
        missing = [name for name in robot.joint_names if name not in by_name]
        if missing:
            raise ValueError(f"observation is missing robot joints {missing}; got {frame.state_names}")
        return state[[by_name[name] for name in robot.joint_names]]

    @staticmethod
    def _state_slices(session: Session, width: int) -> dict[str, tuple[int, int]]:
        declared = session.observation.get("state_split")
        if declared:
            return {str(name): (int(a), int(b)) for name, a, b in declared}
        # Convention across the SO-ARM checkpoints: arm joints then a 1-DOF jaw.
        return {"single_arm": (0, max(0, width - 1)), "gripper": (max(0, width - 1), width)}

    @staticmethod
    def _flatten(chunk: Any) -> np.ndarray:
        """GR00T returns a dict of per-modality chunks; concatenate in order."""
        if isinstance(chunk, np.ndarray):
            return np.atleast_2d(chunk).astype(np.float32)
        parts = [_as_steps(chunk[f"action.{key}"]) for key in MODALITY_KEYS if f"action.{key}" in chunk]
        if not parts:
            raise KeyError(f"no action.* keys in policy output; got {sorted(chunk)}")
        return np.concatenate(parts, axis=-1)


def _as_steps(values: Any) -> np.ndarray:
    """Shape one modality's chunk as ``(steps, width)``.

    A single-width modality such as the gripper arrives 1-D with one entry per
    step. ``np.atleast_2d`` would read that as one step of N values, which then
    fails to concatenate against the arm's N steps.
    """
    array = np.asarray(values, dtype=np.float32)
    return array.reshape(-1, 1) if array.ndim == 1 else np.atleast_2d(array)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="gr00t-n15-server")
    parser.add_argument("--namespace", required=True, help="zenoh key prefix; run.sh passes the workflow name")
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
        format="[gr00t-n15] %(message)s",
    )
    server = Gr00tN15Server(namespace=args.namespace)
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

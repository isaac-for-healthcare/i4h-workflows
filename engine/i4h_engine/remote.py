# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The generic proxy for tasks whose backend lives in another process.

This is **one class for every policy stack**, driven entirely by the manifest.
Adding GR00T N1.8 or a new PI variant means adding a subproject and task
manifest; nothing here changes, and nothing in ``arena`` changes either.

That matters because the arena venv physically cannot import those backends —
their torch pins conflict with Isaac's. The proxy imports nothing from them.

Protocol, per active node (see DESIGN.md §7)::

    on_enter  → publish TaskSpecMsg on  task/{uid}/spec, wait for status=ready
    tick      → publish ObsFrame    on  task/{uid}/obs
                read latest ActionChunk from task/{uid}/action
                write it into ctx.act
                read latest TaskStatusMsg from task/{uid}/status
    on_exit   → report success

Completion is decided by whichever fires first: the backend reporting terminal
status, or a locally-evaluated ``until`` predicate. Both exist because a policy
usually cannot judge its own success but the scene can.
"""

from __future__ import annotations

import logging
import os
import time
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from i4h_common.bus.base import Latest
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import ActionChunk, ObsFrame, TaskSpecMsg, TaskStatusMsg, encode
from i4h_common.manifest import TaskSpec
from i4h_common.types import Pose, satisfied
from i4h_common.world import UnsupportedActuation
from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext

logger = logging.getLogger("i4h.remote")


#: Predicate evaluated locally each tick; returning True ends the node.
UntilPredicate = Callable[[TickContext], Any]

READY_TIMEOUT_ENV = "I4H_BACKEND_READY_TIMEOUT_S"
DEFAULT_READY_TIMEOUT_S = 120.0


def default_ready_timeout_s() -> float:
    """Seconds to wait for a backend, overridable with ``$I4H_BACKEND_READY_TIMEOUT_S``.

    The default suits a warm model cache. A first run downloads the checkpoint
    inside this window, and a multi-GB one does not finish within it.
    """
    override = os.environ.get(READY_TIMEOUT_ENV)
    if not override:
        return DEFAULT_READY_TIMEOUT_S
    try:
        value = float(override)
    except ValueError as exc:
        raise RuntimeError(f"{READY_TIMEOUT_ENV}={override} is not a number") from exc
    if value <= 0.0:
        raise RuntimeError(f"{READY_TIMEOUT_ENV}={override} must be positive")
    return value


class RemoteTaskError(RuntimeError):
    """The backend never appeared, or reported an unrecoverable error."""


@dataclass(slots=True)
class _Outputs:
    success: bool = False


class RemoteTask(Task):
    """Drives one remote inference session for the lifetime of a workflow node."""

    Outputs = _Outputs

    def __init__(
        self,
        spec: TaskSpec,
        *,
        prompt: str = "",
        checkpoint: str = "",
        until: UntilPredicate | None = None,
        max_steps: int | None = None,
        ready_timeout_s: float | None = None,
        action_timeout_s: float = 30.0,
        keys: Keys | None = None,
        name: str | None = None,
        **params: Any,
    ) -> None:
        super().__init__(name=name or spec.name)
        self.spec = spec
        self.prompt = prompt or spec.effective_prompt
        self.checkpoint = checkpoint
        self.until = until
        self.max_steps = max_steps
        self.ready_timeout_s = default_ready_timeout_s() if ready_timeout_s is None else ready_timeout_s
        self.action_timeout_s = action_timeout_s
        self.params = params
        self._keys = keys
        self._uid = ""
        self._actions: Latest[ActionChunk] | None = None
        self._status: Latest[TaskStatusMsg] | None = None
        self._chunk: list[list[float]] = []
        self._chunk_index = 0
        self._succeeded = False
        self._waiting_for_action = False
        self._action_deadline = 0.0
        self._space = ""
        self._layout = ""
        self._robots: tuple[str, ...] = ()
        self._gripper = "none"
        # Own tick counter rather than ctx.node_step: the timeouts below must
        # hold whoever drives this task, including a bare unit test.
        self._ticks = 0

    # -- lifecycle -------------------------------------------------------
    def on_enter(self, ctx: TickContext, inputs: Any) -> None:
        if ctx.bus is None:
            raise RemoteTaskError(
                f"{self.spec.id} is a remote task but no bus is available; "
                f"start the backend with run.sh, or use --no-backend with one already running"
            )
        keys = self._keys or ctx.keys or Keys(ctx.run_id or self.spec.project)
        suffix = f"{ctx.episode_index}-{ctx.attempt_index}" if ctx.attempt_index else str(ctx.episode_index)
        self._uid = f"{self.name}-{suffix}"
        self._actions = Latest(ctx.bus, keys.task_action(self._uid), ActionChunk)
        self._status = Latest(ctx.bus, keys.task_status(self._uid), TaskStatusMsg)
        self._chunk = []
        self._chunk_index = 0
        self._succeeded = False
        self._waiting_for_action = False
        self._action_deadline = 0.0
        self._ticks = 0
        self._ready = False
        self._ready_deadline = 0.0
        self._ready = False

        ctx.bus.publish(
            keys.task_spec(self._uid),
            encode(
                TaskSpecMsg(
                    task_uid=self._uid,
                    task_id=self.spec.id,
                    run_id=ctx.run_id,
                    episode_index=ctx.episode_index,
                    prompt=self.prompt,
                    checkpoint=self.checkpoint,
                    # No model/observation: the backend reads those from its own
                    # catalog. Sending them would make arena carry data it never
                    # looks at, and give the config two places to disagree.
                    params=dict(self.params),
                )
            ),
        )
        self._obs_key = keys.task_obs(self._uid)

        # Waiting happens in tick(), not here. Blocking on_enter stops the
        # runner stepping, which freezes the simulator window until the backend
        # answers — up to ready_timeout_s of an unresponsive UI.
        self._ready = False
        self._ready_deadline = time.monotonic() + self.ready_timeout_s

    def _await_ready(self, ctx: TickContext) -> Status | None:
        """Hold position until the backend answers, one tick at a time.

        Returns ``None`` once ready, so a backend that answered before the
        first tick costs nothing.
        """
        ready = self._status.get() if self._status else None
        if ready is not None and ready.status in ("failure", "error"):
            logger.warning("remote task %s failed during startup: %s", self.spec.id, ready.detail)
            return Status.FAILURE
        if ready is not None and ready.status in ("ready", "running"):
            self._adopt_contract(ready, ctx)
            self._ready = True
            logger.info(
                "remote task %s ready (uid=%s, actions=%s/%s)",
                self.spec.id,
                self._uid,
                self._space,
                self._layout,
            )
            return None
        if time.monotonic() >= self._ready_deadline:
            detail = ready.detail if ready else "no response"
            raise RemoteTaskError(
                f"{self.spec.id}: backend did not become ready within {self.ready_timeout_s}s ({detail}); "
                f"expected a server from {self.spec.backend.project if self.spec.backend else '?'}"
            )
        ctx.act.hold()
        return Status.WAITING

    def _adopt_contract(self, ready: TaskStatusMsg, ctx: TickContext) -> None:
        """Take the backend's word for what its actions mean, then sanity-check it.

        The backend has loaded the checkpoint; the manifest has not. Where they
        disagree the backend wins, but a disagreement with the *scene* is fatal —
        that means the wrong checkpoint is serving this task, and continuing
        would drive the robot with numbers that mean something else.
        """
        declared = self.spec.requires.get("action_space")
        self._space = ready.action_space or declared or ctx.act.action_space
        self._layout = ready.action_layout or ("joints" if self._space != "ee_pose" else "pos_quat")
        self._robots = tuple(ready.action_robots) or tuple(getattr(ctx.scene, "robots", ("robot",)))
        self._gripper = ready.action_gripper or "none"

        if self._space != ctx.act.action_space:
            raise RemoteTaskError(
                f"{self.spec.id}: backend emits {self._space!r} actions but scene accepts "
                f"{ctx.act.action_space!r}. The checkpoint does not match this scene."
            )
        if ready.action_dof and ready.action_dof != ctx.act.dof:
            raise RemoteTaskError(
                f"{self.spec.id}: backend emits {ready.action_dof}-value actions but scene accepts "
                f"{ctx.act.dof}. The checkpoint does not match this scene."
            )
        if declared and ready.action_space and declared != ready.action_space:
            logger.warning(
                "%s: manifest declares action_space=%r but the backend reports %r; using the backend's",
                self.spec.id,
                declared,
                ready.action_space,
            )

    def tick(self, ctx: TickContext) -> Status:
        if ctx.bus is None:
            return Status.FAILURE
        if not self._ready:
            waiting = self._await_ready(ctx)
            if waiting is not None:
                return waiting

        status_msg = self._status.get() if self._status else None
        if status_msg is not None:
            if status_msg.status == "success":
                self._succeeded = True
                return Status.SUCCESS
            if status_msg.status in ("failure", "error"):
                logger.warning("remote task %s failed: %s", self.spec.id, status_msg.detail)
                return Status.FAILURE

        if self.until is not None and satisfied(self.until(ctx)):
            self._succeeded = True
            return Status.SUCCESS

        if self.max_steps is not None and self._ticks >= self.max_steps:
            return Status.FAILURE

        if self._chunk_index >= len(self._chunk) and not self._waiting_for_action:
            ctx.bus.publish(self._obs_key, encode(self._observation(ctx)))
            self._waiting_for_action = True
            self._action_deadline = time.monotonic() + self.action_timeout_s
        action = self._next_action()
        if action is None:
            if time.monotonic() >= self._action_deadline:
                logger.warning("remote task %s produced no action for %.1fs", self.spec.id, self.action_timeout_s)
                return Status.FAILURE
            ctx.act.hold()
            time.sleep(0.001)
            return Status.WAITING

        self._ticks += 1
        self._write(ctx, np.asarray(action, dtype=np.float32))
        return Status.RUNNING

    def on_exit(self, ctx: TickContext) -> _Outputs:
        self._teardown()
        return _Outputs(success=self._succeeded)

    def on_abort(self, ctx: TickContext) -> None:
        self._teardown()

    def _teardown(self) -> None:
        for latest in (self._actions, self._status):
            if latest is not None:
                latest.close()
        self._actions = None
        self._status = None

    # -- helpers ---------------------------------------------------------
    def _observation(self, ctx: TickContext) -> ObsFrame:
        """Build the frame the manifest's ``observation`` block asks for."""
        joints = ctx.scene.joints()
        images: dict[str, bytes] = {}
        shapes: dict[str, list[int]] = {}
        for camera_name in self.spec.requires.get("cameras", ()):
            frame = ctx.scene.camera(camera_name)
            if frame is None:
                continue
            images[camera_name] = frame.data
            shapes[camera_name] = [frame.height, frame.width, 3]
        return ObsFrame(
            task_uid=self._uid,
            step=ctx.node_step,
            state=[float(v) for v in joints.pos[0]],
            state_names=list(joints.names),
            images=images,
            image_shapes=shapes,
        )

    def _next_action(self) -> list[float] | None:
        """Consume the current chunk, refilling from the newest one that arrives."""
        chunk = self._actions.take() if self._actions else None
        if chunk is not None and chunk.dof > 0:
            self._chunk = chunk.reshape()
            self._chunk_index = 0
            self._waiting_for_action = False
        if self._chunk_index < len(self._chunk):
            action = self._chunk[self._chunk_index]
            self._chunk_index += 1
            return action
        return None

    def _write(self, ctx: TickContext, row: np.ndarray) -> None:
        """Apply one action row according to the contract the backend reported."""
        batched = row.reshape(1, -1).repeat(ctx.num_envs, axis=0)
        per_robot = self._split(batched)
        for robot, chunk in per_robot:
            if self._space in ("joint_position", "joint_velocity"):
                if self._gripper == "last" and chunk.shape[-1] > 1:
                    ctx.act.set_joint_targets(chunk, robot)
                else:
                    ctx.act.set_joint_targets(chunk, robot)
            elif self._space == "ee_pose":
                if self._layout == "delta_axis_angle":
                    ctx.act.set_ee_delta(chunk[:, :6], robot)
                else:
                    ctx.act.set_ee_target(self._to_pose(chunk), robot)
                if self._gripper == "last":
                    ctx.act.set_gripper(chunk[:, -1], robot)
            else:
                raise UnsupportedActuation(f"{self.spec.id}: cannot apply action_space={self._space!r}")

    def _split(self, batched: np.ndarray) -> list[tuple[str, np.ndarray]]:
        """Divide one action row across the robots the contract addresses.

        A dual-arm scene gets a 14-wide chunk meaning seven columns per arm; a
        single-arm scene gets the whole row. Splitting here rather than in the
        adapter is what lets the engine's conflict detector see two distinct
        robots being driven, instead of one node writing everything twice.
        """
        robots = self._robots or ("robot",)
        if len(robots) <= 1:
            return [(robots[0] if robots else "robot", batched)]
        width = batched.shape[-1]
        if width % len(robots) != 0:
            raise RemoteTaskError(
                f"{self.spec.id}: action width {width} does not divide evenly across "
                f"{len(robots)} robots {list(robots)}"
            )
        stride = width // len(robots)
        return [(robot, batched[:, i * stride : (i + 1) * stride]) for i, robot in enumerate(robots)]

    def _to_pose(self, chunk: np.ndarray) -> Pose:
        """Interpret a Cartesian action row using the declared layout."""
        pos = chunk[:, :3]
        layout = self._layout
        if layout == "pos_quat":
            if chunk.shape[-1] < 7:
                raise RemoteTaskError(f"{self.spec.id}: pos_quat needs >=7 columns, got {chunk.shape[-1]}")
            return Pose(pos=pos, quat=chunk[:, 3:7])
        if layout in ("pos_euler", "pos_axis_angle"):
            if chunk.shape[-1] < 6:
                raise RemoteTaskError(f"{self.spec.id}: {layout} needs >=6 columns, got {chunk.shape[-1]}")
            rot = chunk[:, 3:6]
            quat = _euler_to_quat(rot) if layout == "pos_euler" else _axis_angle_to_quat(rot)
            return Pose(pos=pos, quat=quat)
        raise RemoteTaskError(
            f"{self.spec.id}: backend reported action_layout={layout!r}, which this proxy does not know. "
            f"Known: pos_quat, pos_euler, pos_axis_angle, delta_axis_angle, joints."
        )

    def describe(self) -> str:
        return f"{self.spec.id} (remote via {self.spec.backend.project if self.spec.backend else '?'})"


def _axis_angle_to_quat(rot: np.ndarray) -> np.ndarray:
    """Batched axis-angle (rotation vector) to ``wxyz`` quaternion."""
    angle = np.linalg.norm(rot, axis=-1, keepdims=True)
    small = angle < 1e-8
    axis = np.divide(rot, np.where(small, 1.0, angle))
    half = angle * 0.5
    quat = np.concatenate([np.cos(half), axis * np.sin(half)], axis=-1)
    identity = np.tile(np.array([1.0, 0.0, 0.0, 0.0], np.float32), (rot.shape[0], 1))
    return np.where(small, identity, quat).astype(np.float32)


def _euler_to_quat(rpy: np.ndarray) -> np.ndarray:
    """Batched intrinsic XYZ euler angles to ``wxyz`` quaternion."""
    half = rpy * 0.5
    cr, cp, cy = np.cos(half[:, 0]), np.cos(half[:, 1]), np.cos(half[:, 2])
    sr, sp, sy = np.sin(half[:, 0]), np.sin(half[:, 1]), np.sin(half[:, 2])
    return np.stack(
        [
            cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy,
        ],
        axis=-1,
    ).astype(np.float32)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared Zenoh runtime primitives.

Each per-env runtime under this package builds on:

* :class:`PolicyIO` — a generic zenoh bridge: publishes camera + state,
  subscribes to action commands, queues them. Chunk-at-a-time replace
  semantics (new chunks are ignored until the previous one drains).
* :func:`run_policy_episode` — the synchronous episode loop. Takes a
  per-env ``publish_obs`` callback for the obs side and a ``policy_action``
  callback that converts a queued action into the per-env env-step input.

Add a new env by writing a small file alongside this one with three things:
its camera keys, a ``publish_obs`` function, and a ``policy_action`` function.
"""

from __future__ import annotations

import faulthandler
import logging
import os
import sys
import threading
import time
from collections import deque
from collections.abc import Callable, Mapping
from contextlib import contextmanager
from typing import Any

import numpy as np
import torch
from common.io.camera import CameraPublisher
from common.io.robot import RobotCommandSubscriber, RobotStatePublisher
from common.messages import CameraStream, RobotCommand, RobotState
from common.zenoh_utils import close_quietly, open_zenoh_session
from tqdm import trange

logger = logging.getLogger("arena")

_REPUBLISH_PERIOD_S = 0.1
_ACTION_WAIT_TIMEOUT_S = 30.0
_ACTION_WAIT_LOG_S = 5.0
_SLOW_PHASE_WARN_S = float(os.environ.get("I4H_ARENA_SLOW_PHASE_WARN_S", "0"))
_PHASE_TRACEBACK_S = float(os.environ.get("I4H_ARENA_PHASE_TRACEBACK_S", "0"))
_POLICY_RENDER_MODE = os.environ.get("I4H_ARENA_POLICY_RENDER_MODE", "full").strip().lower()
if _PHASE_TRACEBACK_S > 0:
    try:
        faulthandler.enable(file=sys.stderr, all_threads=True)
    except Exception:
        logger.debug("failed to enable faulthandler", exc_info=True)


def first_env_rgb(value) -> np.ndarray:
    """Extract a single-env (H, W, 3) uint8 frame from a torch tensor or
    numpy array of shape (B, H, W, C) or (H, W, C)."""
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    return array[..., :3].astype(np.uint8, copy=False)


def find_camera_in_obs(
    observation: Mapping[str, Any],
    keys: str | tuple[str, ...],
    *,
    groups: tuple[str, ...] = ("camera_obs", "policy"),
) -> np.ndarray | None:
    """Locate a camera RGB frame in an Arena observation dict.

    Different observation pathways land in different groups: cameras
    registered by the embodiment expose themselves in
    ``observation["camera_obs"]``, while scene-level cameras surfaced
    through an ``isaaclab.envs.mdp.image`` observation term in the task's
    policy obs group land in ``observation["policy"]``. Walk both so
    per-env runtime publishers don't have to encode this split.

    Returns the frame as a (H, W, 3) uint8 array, or ``None`` when none
    of ``keys`` is found in any of ``groups``.
    """
    if isinstance(keys, str):
        keys = (keys,)
    for group in groups:
        section = observation.get(group, {})
        if not hasattr(section, "get"):
            continue
        for key in keys:
            value = section.get(key)
            if value is not None:
                return first_env_rgb(value)
    return None


class PolicyIO:
    """Zenoh bridge: publishes cameras + state, subscribes to action commands.

    Args:
        camera_keys: ``{cam_label: zenoh_key_expr}``. The labels are what
            ``publish_camera`` callers use; the key expressions are what the
            policy subscribes to.
        state_key: Zenoh key for ``RobotState`` publishes (``None`` → default).
        command_key: Zenoh key for ``RobotCommand`` subscribes (``None`` → default).
        action_dim: width of each row in the action queue (used to reshape
            the flat ``cmd.joint_positions`` array).
        max_execution_steps: cap on how many actions to enqueue from one
            chunk. ``None`` keeps the full chunk.
    """

    def __init__(
        self,
        *,
        camera_keys: Mapping[str, str],
        state_key: str | None = None,
        command_key: str | None = None,
        action_dim: int,
        max_execution_steps: int | None = None,
    ) -> None:
        self._session = open_zenoh_session()
        self._cmd_lock = threading.Lock()
        self._action_queue: deque[np.ndarray] = deque()
        self._action_dim = action_dim
        self._max_execution_steps = max_execution_steps
        self._run_id = ""
        self._episode_index = 0
        self._attempt_index = 1
        self._cameras = {key: CameraPublisher(self._session, expr) for key, expr in camera_keys.items()}
        state_kwargs = {"key_expr": state_key} if state_key else {}
        cmd_kwargs = {"key_expr": command_key} if command_key else {}
        self._state = RobotStatePublisher(self._session, **state_kwargs)
        self._commands = RobotCommandSubscriber(self._session, self._on_command, **cmd_kwargs)

    def __enter__(self) -> "PolicyIO":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        for resource in (self._commands, *self._cameras.values(), self._state, self._session):
            close_quietly(resource)

    def set_run_context(self, *, run_id: str, episode_index: int, attempt_index: int = 1) -> None:
        self._run_id = run_id
        self._episode_index = episode_index
        self._attempt_index = attempt_index
        self.clear_actions()

    def publish_camera(self, cam_key: str, image_rgb: np.ndarray) -> None:
        self._cameras[cam_key].publish(_camera_stream(image_rgb))

    def publish_state(self, joint_positions: np.ndarray) -> None:
        self._state.publish(
            RobotState(
                run_id=self._run_id,
                episode_index=self._episode_index,
                attempt_index=self._attempt_index,
                joint_positions=np.asarray(joint_positions, dtype=float).tolist(),
            )
        )

    def pop_action(self) -> np.ndarray | None:
        with self._cmd_lock:
            return self._action_queue.popleft() if self._action_queue else None

    def has_action(self) -> bool:
        with self._cmd_lock:
            return bool(self._action_queue)

    def action_queue_size(self) -> int:
        with self._cmd_lock:
            return len(self._action_queue)

    def clear_actions(self) -> None:
        with self._cmd_lock:
            self._action_queue.clear()

    def _on_command(self, cmd: RobotCommand) -> None:
        if not self._command_matches_context(cmd):
            logger.debug(
                "ignoring stale policy command: cmd_run=%s/%s/%s current=%s/%s/%s",
                cmd.run_id,
                cmd.episode_index,
                cmd.attempt_index,
                self._run_id,
                self._episode_index,
                self._attempt_index,
            )
            return
        try:
            actions = _decode_command(cmd, self._action_dim)
        except Exception:
            logger.exception("failed to decode policy command")
            return
        with self._cmd_lock:
            # Chunk-at-a-time: only accept a new chunk once the previous one
            # has drained. Diffusion-policy chunks are coherent trajectories;
            # re-planning mid-chunk discards most of that motion.
            if self._action_queue:
                return
            if self._max_execution_steps is not None:
                actions = actions[: self._max_execution_steps]
            for row in actions:
                self._action_queue.append(row.copy())

    def _command_matches_context(self, cmd: RobotCommand) -> bool:
        # Commands must belong to the active episode/attempt. Rejecting
        # untagged commands prevents an old policy daemon from driving a new
        # reset attempt with stale inference output.
        if self._run_id and cmd.run_id != self._run_id:
            return False
        if self._episode_index and cmd.episode_index != self._episode_index:
            return False
        if self._attempt_index and cmd.attempt_index != self._attempt_index:
            return False
        return True


PublishObsFn = Callable[[Any], None]
PolicyActionFn = Callable[[Any], torch.Tensor | None]
SuccessFn = Callable[[Any], bool]


@torch.no_grad()
def run_policy_episode(
    ctx,
    *,
    max_timesteps: int,
    publish_obs: PublishObsFn,
    policy_action: PolicyActionFn,
    success_condition: SuccessFn | None = None,
    stop_on_env_done: bool = False,
) -> str:
    """Synchronous episode loop shared across env runtimes."""
    logger.info("policy episode started (max_timesteps=%s)", max_timesteps)
    ctx.env_terminated = None
    ctx.env_truncated = None
    ctx.policy_step = 0
    ctx.io.clear_actions()
    _apply_policy_render_mode(ctx)
    with _watch_phase(ctx, "publish_obs_initial", -1):
        publish_obs(ctx)
    last_pub = time.monotonic()

    for step in trange(max_timesteps, desc=ctx.env_id, leave=False):
        wait_started = time.monotonic()
        last_wait_log = wait_started
        while not ctx.io.has_action():
            now = time.monotonic()
            if not ctx.simulation_app.is_running() or ctx.controller.should_abort():
                ctx.io.clear_actions()
                return "aborted"
            if now - wait_started > _ACTION_WAIT_TIMEOUT_S:
                ctx.io.clear_actions()
                logger.warning("policy episode timed out waiting for action at step %s", step)
                return "timeout"
            if now - last_pub > _REPUBLISH_PERIOD_S:
                with _watch_phase(ctx, "publish_obs_wait", step):
                    publish_obs(ctx)
                last_pub = now
            if now - last_wait_log > _ACTION_WAIT_LOG_S:
                logger.warning(
                    "waiting for policy action: env=%s step=%s waited=%.1fs queue=%s run=%s episode=%s attempt=%s",
                    ctx.env_id,
                    step,
                    now - wait_started,
                    ctx.io.action_queue_size(),
                    getattr(ctx.io, "_run_id", ""),
                    getattr(ctx.io, "_episode_index", 0),
                    getattr(ctx.io, "_attempt_index", 0),
                )
                last_wait_log = now
            time.sleep(0.01)

        if not ready(ctx):
            ctx.io.clear_actions()
            return "aborted"

        with _watch_phase(ctx, "policy_action", step):
            action = policy_action(ctx)
        if action is None:
            continue
        action_values = action.detach().flatten()
        ctx.policy_action_values = action_values.detach().cpu()
        ctx.policy_action_norm = float(torch.linalg.norm(action_values).item())
        ctx.policy_action_max_abs = float(torch.max(torch.abs(action_values)).item()) if action_values.numel() else 0.0
        ctx.policy_step = step
        with _watch_phase(ctx, "env_step", step):
            _, _, terminated, truncated, _ = ctx.env.step(action.repeat(ctx.env.unwrapped.num_envs, 1))
        ctx.env_terminated = terminated
        ctx.env_truncated = truncated
        ctx.policy_step = step + 1
        if stop_on_env_done and bool(terminated.any().item()):
            ctx.io.clear_actions()
            logger.info("policy episode completed after environment termination")
            return "completed"
        if stop_on_env_done and bool(truncated.any().item()):
            ctx.io.clear_actions()
            logger.info("policy episode ended by environment truncation")
            return "timeout"
        with _watch_phase(ctx, "publish_obs_post_step", step):
            publish_obs(ctx)
        last_pub = time.monotonic()
        if success_condition is not None and success_condition(ctx):
            ctx.io.clear_actions()
            logger.info("policy episode succeeded after success condition was satisfied")
            return "completed"

    ctx.io.clear_actions()
    logger.info("policy episode timed out after %s steps", max_timesteps)
    return "timeout"


@contextmanager
def _watch_phase(ctx, phase: str, step: int):
    start = time.monotonic()
    ctx.policy_phase = phase
    ctx.policy_phase_step = step
    traceback_enabled = _PHASE_TRACEBACK_S > 0 and faulthandler.is_enabled()
    if traceback_enabled:
        faulthandler.dump_traceback_later(
            _PHASE_TRACEBACK_S,
            repeat=False,
            file=sys.stderr,
            exit=False,
        )
    try:
        yield
    finally:
        if traceback_enabled:
            faulthandler.cancel_dump_traceback_later()
        elapsed = time.monotonic() - start
        if _SLOW_PHASE_WARN_S > 0 and elapsed > _SLOW_PHASE_WARN_S:
            logger.warning(
                "slow policy phase: env=%s phase=%s step=%s elapsed=%.3fs queue=%s action_norm=%.4f action_max_abs=%.4f",
                getattr(ctx, "env_id", ""),
                phase,
                step,
                elapsed,
                ctx.io.action_queue_size() if hasattr(ctx, "io") else -1,
                float(getattr(ctx, "policy_action_norm", 0.0)),
                float(getattr(ctx, "policy_action_max_abs", 0.0)),
            )


def _apply_policy_render_mode(ctx) -> None:
    """Switch policy rollouts to partial camera rendering only when explicitly requested."""
    if getattr(ctx, "_policy_render_mode_applied", False):
        return
    ctx._policy_render_mode_applied = True
    if _POLICY_RENDER_MODE in {"", "full", "full_rendering", "viewport"}:
        return
    if _POLICY_RENDER_MODE not in {"partial", "partial_rendering", "cameras"}:
        logger.warning(
            "unknown I4H_ARENA_POLICY_RENDER_MODE=%r; using partial rendering",
            _POLICY_RENDER_MODE,
        )
    sim = getattr(getattr(ctx, "env", None), "sim", None)
    render_modes = getattr(sim, "RenderMode", None)
    set_render_mode = getattr(sim, "set_render_mode", None)
    if render_modes is None or not callable(set_render_mode) or not hasattr(render_modes, "PARTIAL_RENDERING"):
        return
    try:
        current = getattr(sim, "render_mode", None)
        target = render_modes.PARTIAL_RENDERING
        if current != target:
            set_render_mode(target)
            logger.info("policy rollout render mode set to PARTIAL_RENDERING by I4H_ARENA_POLICY_RENDER_MODE")
    except Exception:
        logger.debug("failed to set partial policy render mode", exc_info=True)


def ready(ctx) -> bool:
    while ctx.controller.is_paused():
        if not ctx.simulation_app.is_running() or ctx.controller.should_abort():
            return False
        ctx.env.sim.render()
        time.sleep(0.001)
    return ctx.simulation_app.is_running() and not ctx.controller.should_abort()


def _camera_stream(image_rgb: np.ndarray, *, focal_len: float = 12.0) -> CameraStream:
    height, width = image_rgb.shape[:2]
    return CameraStream(width=width, height=height, focal_len=focal_len, data=image_rgb.astype(np.uint8).tobytes())


def _decode_command(cmd: RobotCommand, action_dim: int) -> np.ndarray:
    if cmd.horizon < 1:
        return np.empty((0, action_dim), dtype=np.float64)
    return np.asarray(cmd.joint_positions, dtype=np.float64).reshape(cmd.horizon, action_dim)

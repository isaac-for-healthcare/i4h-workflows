# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 locomanip Zenoh runtime — tray pick-and-place + cart push."""

from __future__ import annotations

import time
import uuid

import numpy as np
import torch
from arena.runtimes.core.base import PolicyIO, find_camera_in_obs, logger
from common.config import get_policy_config, get_robot_config, get_zenoh_config
from tqdm import trange

# Action-channel slots in the 50D g1_wbc_joint action tensor.
_LOCOMANIP_ACTION_CHANNELS: dict[str, tuple[int, int]] = {
    "joints": (0, 43),
    "navigate_command": (43, 46),
    "base_height_command": (46, 47),
    "torso_orientation_rpy_command": (47, 50),
}

_G1_CONFIG = get_robot_config("g1")
_G1_BODY_JOINT_NAMES = _G1_CONFIG.body_joint_names
_G1_HAND_JOINT_NAMES = _G1_CONFIG.hand_joint_names
_TRAY_LOG_PERIOD_STEPS = 100
_TRAY_PICKED_UP_HEIGHT_M = 0.18
_TRAY_NEAR_TARGET_XY_M = 0.35
_TRAY_SUCCESS_XY_M = 0.35
_TRAY_PLACED_HEIGHT_M = 0.15
_TRAY_DROPPED_Z_M = -0.50
_TRAY_HAND_RELEASE_TOLERANCE_RAD = 0.25
_TRAY_ARM_INIT_TOLERANCE_RAD = 1.0
_TRAY_ARM_INIT_JOINT_NAMES = _G1_BODY_JOINT_NAMES[15:]
_TRAY_ARM_STATE_LOG_PERIOD_STEPS = 25
_TRAY_ARM_STATE_LOG_TOP_K = 6
_PUSH_CART_TARGET_POS = (0.35, -3.30, -0.7875)
_PUSH_CART_MIN_PROGRESS_M = 1.20
_PUSH_CART_SUCCESS_XY_M = 0.50
_PUSH_CART_MAX_Z_ERROR_M = 0.15
_REPUBLISH_PERIOD_S = 0.1
_ACTION_WAIT_TIMEOUT_S = 30.0


class LocomanipPolicyIO(PolicyIO):
    def __init__(self, *, env_id: str) -> None:
        zenoh = get_zenoh_config(env_id)
        super().__init__(
            camera_keys=zenoh.camera_keys,
            state_key=zenoh.robot_state_key,
            command_key=zenoh.robot_command_key,
            action_dim=0,
        )
        self._action_overrides = self._resolve_action_overrides(env_id)
        if self._action_overrides:
            logger.info(
                "locomanip action overrides for env %s: %s",
                env_id,
                {k: f"{v:.4f}" for k, v in self._action_overrides.items()},
            )

    @staticmethod
    def _resolve_action_overrides(env_id: str) -> dict[str, float]:
        """Read ``policy.action_overrides`` from the env YAML and validate the keys.

        Returns a dict mapping channel-name -> scalar value. Unknown
        channel names are dropped with a warning so a typo in the YAML
        doesn't silently rebind the wrong slots.
        """
        try:
            cfg = get_policy_config(env_id)
        except Exception:
            return {}
        overrides_raw = getattr(cfg, "action_overrides", None) or {}
        if not isinstance(overrides_raw, dict):
            logger.warning(
                "policy.action_overrides for env %s must be a mapping; got %s",
                env_id,
                type(overrides_raw).__name__,
            )
            return {}
        out: dict[str, float] = {}
        for name, value in overrides_raw.items():
            if name not in _LOCOMANIP_ACTION_CHANNELS:
                logger.warning(
                    "policy.action_overrides for env %s references unknown channel %r; " "valid channels: %s",
                    env_id,
                    name,
                    sorted(_LOCOMANIP_ACTION_CHANNELS),
                )
                continue
            out[name] = float(value)
        return out

    def publish_observation(self, observation: dict) -> None:
        joints = _robot_joint_positions(observation)
        for cam_label in self._cameras:
            frame = _find_camera_by_label(observation, cam_label)
            if frame is None:
                if cam_label == "head":
                    raise KeyError(
                        f"head camera not found in observation camera keys: "
                        f"{list(observation.get('camera_obs', {}).keys())}"
                    )
                logger.warning(
                    "camera label %r declared in zenoh.camera_names but not in observation; skipping",
                    cam_label,
                )
                continue
            self.publish_camera(cam_label, frame)
        self.publish_state(joints)

    def pop_action(self):  # type: ignore[override]
        action = super().pop_action()
        if action is None or not self._action_overrides:
            return action
        arr = np.asarray(action, dtype=np.float32).copy()
        for name, value in self._action_overrides.items():
            start, end = _LOCOMANIP_ACTION_CHANNELS[name]
            if arr.ndim == 1:
                if end <= arr.shape[0]:
                    arr[start:end] = value
            else:
                if end <= arr.shape[-1]:
                    arr[..., start:end] = value
        return arr

    def _on_command(self, cmd) -> None:  # type: ignore[override]
        if not self._command_matches_context(cmd):
            logger.debug(
                "ignoring stale locomanip command: cmd_run=%s/%s/%s current=%s/%s/%s",
                cmd.run_id,
                cmd.episode_index,
                cmd.attempt_index,
                self._run_id,
                self._episode_index,
                self._attempt_index,
            )
            return
        actions = np.asarray(cmd.joint_positions, dtype=np.float32)
        if cmd.horizon < 1:
            return
        actions = actions.reshape(cmd.horizon, -1)
        with self._cmd_lock:
            for row in actions:
                self._action_queue.append(row.copy())


def run(ctx, args_cli, *, policy_io_cls: type | None = None) -> None:
    """Locomanip rollout loop.

    Per-env runtime modules that fork this env can pass a
    ``LocomanipPolicyIO`` subclass via ``policy_io_cls`` to override the
    Zenoh publisher (extra cameras, different state extraction, etc.)
    without monkeypatching the module.
    """
    if policy_io_cls is None:
        policy_io_cls = LocomanipPolicyIO
    max_timesteps = args_cli.max_timesteps or 2000
    save_episode_cb = getattr(ctx, "save_episode_cb", None)
    save_successful_episode_cb = getattr(ctx, "save_successful_episode_cb", None) or save_episode_cb
    discard_episode_cb = getattr(ctx, "discard_episode_cb", None)
    save_all = bool(getattr(ctx, "save_all_episodes", False))
    if args_cli.max_attempts and args_cli.max_attempts > 0:
        max_attempts_per_episode = args_cli.max_attempts
    else:
        max_attempts_per_episode = 1
    progress_label = "saved" if save_episode_cb is not None else "completed"
    policy_job_id = uuid.uuid4().hex[:8]
    logger.info(
        "Locomanip policy job: env=%s episodes=%s max_timesteps=%s save_all=%s max_attempts_per_episode=%s",
        ctx.env_id,
        args_cli.episodes,
        max_timesteps,
        save_all,
        max_attempts_per_episode,
    )
    with policy_io_cls(env_id=ctx.env_id) as io:
        ctx.io = io
        saved = 0
        failed = 0
        total_attempts = 0
        episode_attempts = 0
        current_episode = 1
        while current_episode <= args_cli.episodes:
            if not ctx.simulation_app.is_running() or ctx.controller.should_abort():
                break
            if episode_attempts >= max_attempts_per_episode:
                logger.info(
                    "Locomanip policy episode %s/%s exhausted %s attempts; marking failed and moving to next requested episode",
                    current_episode,
                    args_cli.episodes,
                    max_attempts_per_episode,
                )
                failed += 1
                current_episode += 1
                episode_attempts = 0
                continue
            episode_attempts += 1
            total_attempts += 1
            io.clear_actions()
            obs, _ = ctx.env.reset()
            run_id = f"{ctx.env_id}-{policy_job_id}-episode-{current_episode:03d}-attempt-{episode_attempts:02d}"
            io.set_run_context(run_id=run_id, episode_index=current_episode, attempt_index=episode_attempts)
            logger.info(
                "Locomanip policy episode %s/%s attempt %s/%s started; %s=%s/%s",
                current_episode,
                args_cli.episodes,
                episode_attempts,
                max_attempts_per_episode,
                progress_label,
                saved,
                args_cli.episodes,
            )
            status = _run_episode(
                ctx, io, obs, max_timesteps=max_timesteps, progress=_progress_tracker(ctx.env_id, ctx.env)
            )
            completed_episode = current_episode
            completed_attempt = episode_attempts
            metadata = {
                "env_id": ctx.env_id,
                "run_id": run_id,
                "episode_index": completed_episode,
                "attempt_index": completed_attempt,
                "status": status,
            }
            is_success = status == "completed"
            if save_episode_cb is not None:
                if is_success:
                    assert save_successful_episode_cb is not None
                    save_successful_episode_cb(metadata)
                    saved += 1
                    current_episode += 1
                    episode_attempts = 0
                    ctx.controller.episode_completed()
                elif save_all:
                    save_episode_cb(metadata)
                    saved += 1
                    failed += 1
                    current_episode += 1
                    episode_attempts = 0
                elif discard_episode_cb is not None:
                    discard_episode_cb()
            elif is_success:
                saved += 1
                current_episode += 1
                episode_attempts = 0
                ctx.controller.episode_completed()
            logger.info(
                "Locomanip policy episode %s/%s attempt %s/%s: %s",
                completed_episode,
                args_cli.episodes,
                completed_attempt,
                max_attempts_per_episode,
                status,
            )
        logger.info(
            "Locomanip policy job complete: env=%s %s=%s/%s failed=%s total_attempts=%s",
            ctx.env_id,
            progress_label,
            saved,
            args_cli.episodes,
            failed,
            total_attempts,
        )


@torch.no_grad()
def _run_episode(ctx, io, obs: dict, *, max_timesteps: int, progress=None) -> str:
    if progress is not None:
        progress.log_start()
    for step in trange(max_timesteps, desc=ctx.env_id, leave=False):
        if not _ready(ctx):
            return "aborted"
        io.publish_observation(obs)
        action = _wait_for_action(ctx, io, obs)
        if action is None:
            return "timeout"
        action_tensor = torch.as_tensor(action, device=ctx.env.device, dtype=torch.float32)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.shape[0] == 1 and ctx.env.num_envs > 1:
            action_tensor = action_tensor.repeat(ctx.env.num_envs, 1)
        obs, _, terminated, truncated, _ = ctx.env.step(action_tensor)
        done = bool((terminated | truncated).any().item())
        success = bool(_success_terminated(ctx.env))
        if success:
            if progress is not None:
                progress.log_end("success")
            return "completed"
        if done:
            if progress is not None:
                if progress.complete_now():
                    progress.log_end("fallback_success")
                    logger.info("%s success: %s", ctx.env_id, progress.success_reason())
                    return "completed"
                progress.log_end("terminated_without_success")
            reasons = _termination_reasons(ctx.env)
            if reasons:
                logger.info("%s termination reason(s): %s", ctx.env_id, ", ".join(reasons))
            return "timeout"
        if progress is not None:
            progress.update(step + 1)
            if progress.complete_now():
                progress.log_end("fallback_success")
                logger.info("%s success: %s", ctx.env_id, progress.success_reason())
                return "completed"
    if progress is not None:
        progress.log_end("timeout")
        if progress.success_at_timeout():
            logger.info("%s success: %s", ctx.env_id, progress.success_reason())
            return "completed"
    return "timeout"


def _wait_for_action(ctx, io, obs: dict) -> np.ndarray | None:
    wait_started = time.monotonic()
    last_pub = wait_started
    while _ready(ctx):
        action = io.pop_action()
        if action is not None:
            return action
        now = time.monotonic()
        if now - wait_started > _ACTION_WAIT_TIMEOUT_S:
            logger.warning(
                "timed out waiting for current-run policy action; restart the policy daemon if stale commands were rejected"
            )
            io.clear_actions()
            return None
        if now - last_pub > _REPUBLISH_PERIOD_S:
            io.publish_observation(obs)
            last_pub = now
        time.sleep(0.001)
    return None


def _ready(ctx) -> bool:
    return ctx.simulation_app.is_running() and not ctx.controller.should_abort()


def _success_terminated(env) -> bool:
    try:
        terminations = env.unwrapped.termination_manager.get_active_iterable_terms(env)
        return any(name == "success" and bool(value[0].item()) for name, value in terminations)
    except Exception:
        return False


def _termination_reasons(env) -> list[str]:
    try:
        terminations = env.unwrapped.termination_manager.get_active_iterable_terms(env)
        return [name for name, value in terminations if bool(value[0].item())]
    except Exception:
        return []


def _progress_tracker(env_id: str, env):
    if env_id == "locomanip_tray_pick_and_place":
        return TrayPickPlaceProgress(env)
    if env_id == "locomanip_push_cart":
        return PushCartProgress(env)
    return None


class TrayPickPlaceProgress:
    def __init__(self, env) -> None:
        self._env = env
        self._initial_tray_pos: torch.Tensor | None = None
        self._target_pos: torch.Tensor | None = None
        self._picked = False
        self._near_target = False
        self._placing = False
        self._grip_observed = False
        self._released = False
        self._arms_returned = False
        self._max_lift_m = 0.0
        self._min_xy_distance_m = float("inf")
        self._last_hand_open_error_rad = float("inf")
        self._min_hand_open_error_rad = float("inf")
        self._max_hand_open_error_rad = 0.0
        self._last_arm_init_error_rad = float("inf")
        self._last_tray_pos: torch.Tensor | None = None
        self._last_target_pos: torch.Tensor | None = None
        self._last_lift_m = 0.0
        self._last_xy_distance_m = float("inf")

    def log_start(self) -> None:
        tray_pos, target_pos = self._positions()
        self._initial_tray_pos = tray_pos
        self._target_pos = target_pos
        logger.info(
            "tray pick/place target: tray=%s target=%s xy_distance_m=%.3f",
            _format_vec3(tray_pos),
            _format_vec3(target_pos),
            _xy_distance(tray_pos, target_pos),
        )

    def update(self, step: int) -> None:
        tray_pos, target_pos = self._positions()
        initial_z = (
            float(self._initial_tray_pos[2].item()) if self._initial_tray_pos is not None else float(tray_pos[2].item())
        )
        lift_m = float(tray_pos[2].item()) - initial_z
        xy_distance_m = _xy_distance(tray_pos, target_pos)
        self._last_tray_pos = tray_pos
        self._last_target_pos = target_pos
        self._last_lift_m = lift_m
        self._last_xy_distance_m = xy_distance_m
        self._max_lift_m = max(self._max_lift_m, lift_m)
        self._min_xy_distance_m = min(self._min_xy_distance_m, xy_distance_m)
        hands_open, hand_open_error_rad = self._hands_open()
        self._last_hand_open_error_rad = hand_open_error_rad
        self._min_hand_open_error_rad = min(self._min_hand_open_error_rad, hand_open_error_rad)
        self._max_hand_open_error_rad = max(self._max_hand_open_error_rad, hand_open_error_rad)
        arms_in_init, arm_init_error_rad, arm_offsets = self._arms_in_init_pose()
        self._last_arm_init_error_rad = arm_init_error_rad
        if self._picked and not self._grip_observed and hand_open_error_rad > _TRAY_HAND_RELEASE_TOLERANCE_RAD:
            self._grip_observed = True
            logger.info(
                "tray milestone: step=%s grip observed (hand_open_error_rad=%.3f threshold=%.3f)",
                step,
                hand_open_error_rad,
                _TRAY_HAND_RELEASE_TOLERANCE_RAD,
            )
        released_near_target = (
            self._picked and xy_distance_m <= _TRAY_SUCCESS_XY_M and self._grip_observed and hands_open
        )

        if not self._picked and lift_m >= _TRAY_PICKED_UP_HEIGHT_M:
            self._picked = True
            logger.info(
                "tray milestone: step=%s picked up tray (tray=%s lift_m=%.3f)", step, _format_vec3(tray_pos), lift_m
            )
        if self._picked and not self._near_target and xy_distance_m <= _TRAY_NEAR_TARGET_XY_M:
            self._near_target = True
            logger.info(
                "tray milestone: step=%s carried tray near target (tray=%s target=%s xy_distance_m=%.3f)",
                step,
                _format_vec3(tray_pos),
                _format_vec3(target_pos),
                xy_distance_m,
            )
        if self._near_target and not self._placing and lift_m <= _TRAY_PLACED_HEIGHT_M:
            self._placing = True
            logger.info(
                "tray milestone: step=%s lowered tray for placement (tray=%s lift_m=%.3f threshold=%.3f)",
                step,
                _format_vec3(tray_pos),
                lift_m,
                _TRAY_PLACED_HEIGHT_M,
            )
        if released_near_target and not self._released:
            self._released = True
            logger.info(
                "tray milestone: step=%s released grip (hand_open_error_rad=%.3f threshold=%.3f)",
                step,
                hand_open_error_rad,
                _TRAY_HAND_RELEASE_TOLERANCE_RAD,
            )
        if released_near_target and arms_in_init and not self._arms_returned:
            self._arms_returned = True
            logger.info(
                "tray milestone: step=%s arms returned to init pose (arm_init_error_rad=%.3f threshold=%.3f)",
                step,
                arm_init_error_rad,
                _TRAY_ARM_INIT_TOLERANCE_RAD,
            )
        if self._released and (not self._arms_returned) and (step <= 1 or step % _TRAY_ARM_STATE_LOG_PERIOD_STEPS == 0):
            logger.info(
                "tray arm-return state: step=%s arm_init_error_rad=%.3f threshold=%.3f worst_offsets=%s",
                step,
                arm_init_error_rad,
                _TRAY_ARM_INIT_TOLERANCE_RAD,
                _format_joint_offsets(arm_offsets),
            )
        if step <= 1 or step % _TRAY_LOG_PERIOD_STEPS == 0:
            logger.info(
                "tray trajectory: step=%s tray=%s target=%s lift_m=%.3f xy_distance_m=%.3f "
                "picked=%s near_target=%s placing=%s grip_observed=%s released=%s arms_returned=%s "
                "hand_open_error_rad=%.3f arm_init_error_rad=%.3f",
                step,
                _format_vec3(tray_pos),
                _format_vec3(target_pos),
                lift_m,
                xy_distance_m,
                self._picked,
                self._near_target,
                self._placing,
                self._grip_observed,
                self._released,
                self._arms_returned,
                hand_open_error_rad,
                arm_init_error_rad,
            )

    def log_end(self, status: str) -> None:
        tray_pos, target_pos = self._last_positions()
        logger.info(
            "tray %s state: tray=%s target=%s lift_m=%.3f xy_distance_m=%.3f "
            "placed_height_threshold=%.3f success_xy_threshold=%.3f "
            "max_lift_m=%.3f min_xy_distance_m=%.3f picked=%s near_target=%s placing=%s "
            "grip_observed=%s released=%s arms_returned=%s hand_open_error_rad=%.3f "
            "min_hand_open_error_rad=%.3f max_hand_open_error_rad=%.3f "
            "arm_init_error_rad=%.3f arm_init_threshold=%.3f",
            status,
            _format_vec3(tray_pos),
            _format_vec3(target_pos),
            self._last_lift_m,
            self._last_xy_distance_m,
            _TRAY_PLACED_HEIGHT_M,
            _TRAY_SUCCESS_XY_M,
            self._max_lift_m,
            self._min_xy_distance_m,
            self._picked,
            self._near_target,
            self._placing,
            self._grip_observed,
            self._released,
            self._arms_returned,
            self._last_hand_open_error_rad,
            self._min_hand_open_error_rad,
            self._max_hand_open_error_rad,
            self._last_arm_init_error_rad,
            _TRAY_ARM_INIT_TOLERANCE_RAD,
        )

    def success_at_timeout(self) -> bool:
        return self.complete_now() or (
            self._released
            and self._last_xy_distance_m <= _TRAY_SUCCESS_XY_M
            and self._last_tray_z_m() > _TRAY_DROPPED_Z_M
        )

    def complete_now(self) -> bool:
        return self._released and self._arms_returned

    def success_reason(self) -> str:
        if self._arms_returned:
            return "native success did not fire, but tray was released near the cart and the arms returned to init pose"
        return "native success did not fire, but timeout ended with tray released near the cart and not dropped"

    def _positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        scene = self._env.unwrapped.scene
        tray_pos = scene["surgical_tray"].data.root_pos_w[0].detach().cpu()
        target_pos = scene["cart"].data.root_pos_w[0].detach().cpu()
        return tray_pos, target_pos

    def _last_positions(self) -> tuple[torch.Tensor, torch.Tensor]:
        if self._last_tray_pos is not None and self._last_target_pos is not None:
            return self._last_tray_pos, self._last_target_pos
        return self._positions()

    def _last_tray_z_m(self) -> float:
        tray_pos, _ = self._last_positions()
        return float(tray_pos[2].item())

    def _hands_open(self) -> tuple[bool, float]:
        env = self._env.unwrapped
        robot = env.scene["robot"]
        joint_ids = _joint_ids(robot.data.joint_names, _G1_HAND_JOINT_NAMES, env.device)
        current = robot.data.joint_pos[0, joint_ids]
        target = robot.data.default_joint_pos[0, joint_ids]
        max_error = float(torch.max(torch.abs(current - target)).detach().cpu().item())
        return max_error <= _TRAY_HAND_RELEASE_TOLERANCE_RAD, max_error

    def _arms_in_init_pose(self) -> tuple[bool, float, list[tuple[str, float, float, float]]]:
        env = self._env.unwrapped
        robot = env.scene["robot"]
        joint_ids = _joint_ids(robot.data.joint_names, _TRAY_ARM_INIT_JOINT_NAMES, env.device)
        current = robot.data.joint_pos[0, joint_ids]
        target = robot.data.default_joint_pos[0, joint_ids]
        errors = torch.abs(current - target)
        max_error = float(torch.max(errors).detach().cpu().item())
        current_cpu = current.detach().cpu().tolist()
        target_cpu = target.detach().cpu().tolist()
        errors_cpu = errors.detach().cpu().tolist()
        offsets = sorted(
            zip(_TRAY_ARM_INIT_JOINT_NAMES, current_cpu, target_cpu, errors_cpu),
            key=lambda item: item[3],
            reverse=True,
        )
        return max_error <= _TRAY_ARM_INIT_TOLERANCE_RAD, max_error, offsets


class PushCartProgress:
    def __init__(self, env) -> None:
        self._env = env
        self._initial_cart_pos: torch.Tensor | None = None
        self._target_pos = torch.tensor(_PUSH_CART_TARGET_POS, dtype=torch.float32)
        self._last_cart_pos: torch.Tensor | None = None
        self._last_y_progress_m = 0.0
        self._last_xy_distance_m = float("inf")
        self._max_y_progress_m = 0.0
        self._min_xy_distance_m = float("inf")
        self._pushed = False
        self._near_target = False

    def log_start(self) -> None:
        cart_pos = self._cart_pos()
        self._initial_cart_pos = cart_pos
        logger.info(
            "push cart target: cart=%s target=%s xy_distance_m=%.3f",
            _format_vec3(cart_pos),
            _format_vec3(self._target_pos),
            _xy_distance(cart_pos, self._target_pos),
        )

    def update(self, step: int) -> None:
        cart_pos = self._cart_pos()
        initial_y = (
            float(self._initial_cart_pos[1].item()) if self._initial_cart_pos is not None else float(cart_pos[1].item())
        )
        y_progress_m = initial_y - float(cart_pos[1].item())
        xy_distance_m = _xy_distance(cart_pos, self._target_pos)
        self._last_cart_pos = cart_pos
        self._last_y_progress_m = y_progress_m
        self._last_xy_distance_m = xy_distance_m
        self._max_y_progress_m = max(self._max_y_progress_m, y_progress_m)
        self._min_xy_distance_m = min(self._min_xy_distance_m, xy_distance_m)

        if not self._pushed and y_progress_m >= 0.50:
            self._pushed = True
            logger.info(
                "push cart milestone: step=%s cart moved (cart=%s y_progress_m=%.3f)",
                step,
                _format_vec3(cart_pos),
                y_progress_m,
            )
        if not self._near_target and self._within_fallback_success(cart_pos, y_progress_m):
            self._near_target = True
            logger.info(
                "push cart milestone: step=%s cart reached fallback target (cart=%s target=%s y_progress_m=%.3f xy_distance_m=%.3f)",
                step,
                _format_vec3(cart_pos),
                _format_vec3(self._target_pos),
                y_progress_m,
                xy_distance_m,
            )
        if step <= 1 or step % _TRAY_LOG_PERIOD_STEPS == 0:
            logger.info(
                "push cart trajectory: step=%s cart=%s target=%s y_progress_m=%.3f xy_distance_m=%.3f pushed=%s near_target=%s",
                step,
                _format_vec3(cart_pos),
                _format_vec3(self._target_pos),
                y_progress_m,
                xy_distance_m,
                self._pushed,
                self._near_target,
            )

    def log_end(self, status: str) -> None:
        cart_pos = self._last_cart_pos if self._last_cart_pos is not None else self._cart_pos()
        logger.info(
            "push cart %s state: cart=%s target=%s y_progress_m=%.3f xy_distance_m=%.3f "
            "max_y_progress_m=%.3f min_xy_distance_m=%.3f pushed=%s near_target=%s",
            status,
            _format_vec3(cart_pos),
            _format_vec3(self._target_pos),
            self._last_y_progress_m,
            self._last_xy_distance_m,
            self._max_y_progress_m,
            self._min_xy_distance_m,
            self._pushed,
            self._near_target,
        )

    def complete_now(self) -> bool:
        """Live check the per-step driver calls between updates. Mirrors
        TrayPickPlaceProgress.complete_now's contract: true when the agent
        currently satisfies the fallback success criteria."""
        cart_pos = self._last_cart_pos if self._last_cart_pos is not None else self._cart_pos()
        return self._within_fallback_success(cart_pos, self._last_y_progress_m)

    def success_at_timeout(self) -> bool:
        cart_pos = self._last_cart_pos if self._last_cart_pos is not None else self._cart_pos()
        return self._within_fallback_success(cart_pos, self._max_y_progress_m)

    def success_reason(self) -> str:
        return "native success did not fire, but cart reached push-cart fallback criteria"

    def _within_fallback_success(self, cart_pos: torch.Tensor, y_progress_m: float) -> bool:
        return (
            y_progress_m >= _PUSH_CART_MIN_PROGRESS_M
            and _xy_distance(cart_pos, self._target_pos) <= _PUSH_CART_SUCCESS_XY_M
            and abs(float(cart_pos[2].item()) - _PUSH_CART_TARGET_POS[2]) <= _PUSH_CART_MAX_Z_ERROR_M
        )

    def _cart_pos(self) -> torch.Tensor:
        env = self._env.unwrapped
        cart_pos = env.scene["cart"].data.root_pos_w[0] - env.scene.env_origins[0]
        return cart_pos.detach().cpu()


def _xy_distance(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(torch.linalg.norm(a[:2] - b[:2]).item())


def _joint_ids(joint_names: list[str], selected_names: tuple[str, ...], device: str) -> torch.Tensor:
    name_to_index = {name: index for index, name in enumerate(joint_names)}
    return torch.tensor([name_to_index[name] for name in selected_names], device=device, dtype=torch.long)


def _format_vec3(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values.detach().cpu().flatten()[:3].tolist()) + "]"


def _format_joint_offsets(offsets: list[tuple[str, float, float, float]]) -> str:
    return ", ".join(
        f"{name}:cur={current:.3f},init={target:.3f},err={error:.3f}"
        for name, current, target, error in offsets[:_TRAY_ARM_STATE_LOG_TOP_K]
    )


def _head_camera_rgb(observation: dict) -> np.ndarray:
    head = _find_camera_by_label(observation, "head")
    if head is None:
        camera_obs = observation.get("camera_obs", {})
        raise KeyError(f"could not find head camera in observation camera keys: {list(camera_obs.keys())}")
    return head


def _find_camera_by_label(observation: dict, cam_label: str) -> np.ndarray | None:
    """Resolve a Zenoh-side camera label to the matching observation frame.

    Tries the naming conventions Arena's task/embodiment cfgs use in order:
    ``robot_<label>_cam_rgb`` (default Arena ObsTerm), ``robot_<label>_cam``
    (sometimes used when the cfg name lacks _rgb), ``<label>_cam_rgb`` /
    ``<label>_cam`` (scene-cfg local naming), then the bare ``<label>``. For
    head specifically, also accepts the legacy "rgb" key.
    """
    candidates = [
        f"robot_{cam_label}_cam_rgb",
        f"robot_{cam_label}_cam",
        f"{cam_label}_cam_rgb",
        f"{cam_label}_cam",
        cam_label,
    ]
    if cam_label == "head":
        candidates.append("rgb")
    return find_camera_in_obs(observation, tuple(candidates))


def _robot_joint_positions(observation: dict) -> np.ndarray:
    policy_obs = observation.get("policy", {})
    for key in ("robot_joint_pos", "joint_pos", "robot_joints"):
        if key in policy_obs:
            values = policy_obs[key]
            if hasattr(values, "detach"):
                values = values.detach().cpu().numpy()
            values = np.asarray(values, dtype=np.float32)
            return values[0] if values.ndim > 1 else values
    raise KeyError(f"could not find joint positions in policy observation keys: {list(policy_obs.keys())}")

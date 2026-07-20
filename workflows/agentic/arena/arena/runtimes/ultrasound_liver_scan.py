# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka ultrasound liver-scan Zenoh runtime.

The Franka arena env's action space is a
``DifferentialInverseKinematicsActionCfg`` with ``command_type='pose'`` and
``use_relative_mode=True`` — 6 floats per step (dx, dy, dz, droll, dpitch,
dyaw). PI0's chunked output is sliced to 6 dims, so the wire format aligns.
"""

from __future__ import annotations

import math
import time

import numpy as np
import torch
from arena.runtimes.core.base import PolicyIO, logger, run_policy_episode
from common.config import get_zenoh_config

FRANKA_STATE_DIM = 7
ULTRASOUND_ACTION_DIM = 6
DEFAULT_TARGET_TOLERANCE_M = 0.20
DEFAULT_ALIGNMENT_THRESHOLD = 0.80
DEFAULT_MIN_SCAN_STEPS = 180
DEFAULT_ACTION_SATURATION_THRESHOLD = 0.02
DEFAULT_ACTION_SATURATION_STEPS = 20
_TRAJECTORY_LOG_PERIOD_STEPS = 25
_CONTACT_Z_MAX_M = 0.22
_TWIST_ALIGNMENT_THRESHOLD = 0.95
_SWIPE_START_DISTANCE_M = 0.03
_SCAN_COMPLETE_DISTANCE_M = 0.10

# Home/SETUP pose used by workflows/robotic_ultrasound sim_with_dds.py. The
# joint-default init_state lands the arm tucked above the base, but the PI0
# policy was trained from this SETUP pose hovering over the phantom. Without
# pre-rolling the arm here, the first observation is off-distribution and the
# model drifts. Constants mirror workflows/robotic_ultrasound state_machine/utils.py.
_SETUP_POS = (0.3229, -0.0110, 0.3000)
_DOWN_EULER_DEG = (180.0, 0.0, 180.0)
_RESET_STEPS = 40


class UltrasoundPolicyIO(PolicyIO):
    def __init__(self, *, env_id: str) -> None:
        zenoh = get_zenoh_config(env_id)
        # PI0 emits 50-step relative-pose chunks but was trained for receding-
        # horizon control. The reference impl (workflows/robotic_ultrasound
        # sim_with_dds.py) consumes only the first ``replan_steps=5`` of each
        # chunk before re-querying. Anything larger lets the model drift off
        # its training distribution and the arm freezes mid-trajectory.
        super().__init__(
            camera_keys=zenoh.camera_keys,
            state_key=zenoh.robot_state_key,
            command_key=zenoh.robot_command_key,
            action_dim=ULTRASOUND_ACTION_DIM,
            max_execution_steps=5,
        )


def run_policy_based_episode(
    ctx,
    *,
    max_timesteps: int,
    success_target_tolerance_m: float = DEFAULT_TARGET_TOLERANCE_M,
    success_alignment_threshold: float = DEFAULT_ALIGNMENT_THRESHOLD,
    success_min_scan_steps: int = DEFAULT_MIN_SCAN_STEPS,
    success_action_saturation_threshold: float = DEFAULT_ACTION_SATURATION_THRESHOLD,
    success_action_saturation_steps: int = DEFAULT_ACTION_SATURATION_STEPS,
) -> str:
    success_condition = UltrasoundSuccessTracker(
        target_tolerance_m=success_target_tolerance_m,
        alignment_threshold=success_alignment_threshold,
        min_scan_steps=success_min_scan_steps,
        action_saturation_threshold=success_action_saturation_threshold,
        action_saturation_steps=success_action_saturation_steps,
    )
    success_condition.log_target(ctx)

    status = run_policy_episode(
        ctx,
        max_timesteps=max_timesteps,
        publish_obs=publish_obs,
        policy_action=_policy_action,
        success_condition=success_condition,
    )
    if status == "timeout":
        success_condition.log_timeout(ctx)
        if _success_terminated(ctx):
            logger.info("ultrasound success: completed by environment success termination at episode horizon")
            return "completed"
    return status


class UltrasoundSuccessTracker:
    def __init__(
        self,
        *,
        target_tolerance_m: float,
        alignment_threshold: float,
        min_scan_steps: int,
        action_saturation_threshold: float,
        action_saturation_steps: int,
    ) -> None:
        self._target_tolerance_m = target_tolerance_m
        self._alignment_threshold = alignment_threshold
        self._min_scan_steps = min_scan_steps
        self._action_saturation_threshold = action_saturation_threshold
        self._action_saturation_steps = action_saturation_steps
        self._saturated_steps = 0
        self._near_target = False
        self._contact_pos: torch.Tensor | None = None
        self._twist_pos: torch.Tensor | None = None
        self._max_swipe_distance_m = 0.0
        self._seen: set[str] = set()
        self._started_at = time.monotonic()

    def __call__(self, ctx) -> bool:
        metrics = _success_metrics(ctx.env)
        step = _policy_step(ctx)
        action_max_abs = _policy_action_max_abs(ctx)
        self._update_milestones(ctx, metrics)
        self._log_trajectory(ctx, metrics)
        if metrics.distance_m <= self._target_tolerance_m and not self._near_target:
            self._near_target = True
            self._mark(
                "near_target",
                "ultrasound success: step=%s probe reached scan target " "(distance_m=%.3f target=%s ee=%s)",
                _policy_step(ctx),
                metrics.distance_m,
                _format_tensor(metrics.target_pos),
                _format_tensor(metrics.ee_pos),
            )
        elif metrics.distance_m > self._target_tolerance_m:
            self._near_target = False

        if _success_terminated(ctx) or self._scan_goal_satisfied(metrics, step):
            self.log_success(ctx)
            return True

        if step >= self._min_scan_steps and metrics.alignment >= self._alignment_threshold:
            if action_max_abs <= self._action_saturation_threshold:
                self._saturated_steps += 1
            else:
                self._saturated_steps = 0
            if self._saturated_steps == 1:
                logger.info(
                    "ultrasound progress: step=%s action saturation started "
                    "(action_max_abs=%.4f threshold=%.4f min_scan_steps=%s)",
                    step,
                    action_max_abs,
                    self._action_saturation_threshold,
                    self._min_scan_steps,
                )
            if self._saturated_steps == self._action_saturation_steps:
                logger.info(
                    "ultrasound progress: step=%s policy action saturated before scan goal "
                    "(phase=%s held_steps=%s action_max_abs=%.4f threshold=%.4f alignment=%.3f "
                    "swipe_distance_m=%.3f distance_m=%.3f target_tolerance_m=%.3f elapsed_s=%.1f)",
                    step,
                    self._phase(),
                    self._saturated_steps,
                    action_max_abs,
                    self._action_saturation_threshold,
                    metrics.alignment,
                    self._max_swipe_distance_m,
                    metrics.distance_m,
                    self._target_tolerance_m,
                    time.monotonic() - self._started_at,
                )
        else:
            self._saturated_steps = 0

        if metrics.distance_m <= self._target_tolerance_m:
            self._mark(
                "wait_alignment",
                "ultrasound success: step=%s waiting for probe alignment "
                "(distance_m=%.3f alignment=%.3f threshold=%.3f)",
                _policy_step(ctx),
                metrics.distance_m,
                metrics.alignment,
                self._alignment_threshold,
            )
        return False

    def log_target(self, ctx) -> None:
        metrics = _success_metrics(ctx.env)
        logger.info(
            "ultrasound success target: step=0 target=%s ee=%s distance_m=%.3f "
            "target_tolerance_m=%.3f alignment_threshold=%.3f min_scan_steps=%s "
            "action_saturation_threshold=%.4f action_saturation_steps=%s",
            _format_tensor(metrics.target_pos),
            _format_tensor(metrics.ee_pos),
            metrics.distance_m,
            self._target_tolerance_m,
            self._alignment_threshold,
            self._min_scan_steps,
            self._action_saturation_threshold,
            self._action_saturation_steps,
        )

    def log_timeout(self, ctx) -> None:
        state = self._success_state(ctx)
        logger.info(
            "ultrasound success timeout state: step=%s success=%s near_target=%s scan_complete=%s "
            "contact_seen=%s twist_seen=%s target=%s ee=%s distance_m=%.3f alignment=%.3f phase=%s "
            "scan_distance_m=%.3f action_max_abs=%.4f saturated_steps=%s target_tolerance_m=%.3f "
            "alignment_threshold=%.3f scan_distance_threshold_m=%.3f",
            _policy_step(ctx),
            state["success"],
            state["near_target"],
            state["scan_complete"],
            state["contact_seen"],
            state["twist_seen"],
            _format_tensor(state["target_pos"]),
            _format_tensor(state["ee_pos"]),
            state["distance_m"],
            state["alignment"],
            self._phase(),
            state["scan_distance_m"],
            _policy_action_max_abs(ctx),
            self._saturated_steps,
            state["target_tolerance_m"],
            state["alignment_threshold"],
            state["scan_distance_threshold_m"],
        )

    def log_success(self, ctx) -> None:
        state = self._success_state(ctx)
        logger.info(
            "ultrasound success final state: step=%s success=%s near_target=%s scan_complete=%s "
            "contact_seen=%s twist_seen=%s target=%s ee=%s distance_m=%.3f alignment=%.3f phase=%s "
            "scan_distance_m=%.3f elapsed_s=%.1f target_tolerance_m=%.3f alignment_threshold=%.3f "
            "scan_distance_threshold_m=%.3f",
            _policy_step(ctx),
            state["success"],
            state["near_target"],
            state["scan_complete"],
            state["contact_seen"],
            state["twist_seen"],
            _format_tensor(state["target_pos"]),
            _format_tensor(state["ee_pos"]),
            state["distance_m"],
            state["alignment"],
            self._phase(),
            state["scan_distance_m"],
            time.monotonic() - self._started_at,
            state["target_tolerance_m"],
            state["alignment_threshold"],
            state["scan_distance_threshold_m"],
        )

    def scan_goal_satisfied(self, ctx) -> bool:
        return _success_terminated(ctx)

    def _update_milestones(self, ctx, metrics: "_SuccessMetrics") -> None:
        step = _policy_step(ctx)
        if self._contact_pos is None and metrics.ee_pos[2].item() <= _CONTACT_Z_MAX_M:
            self._contact_pos = metrics.ee_pos.clone()
            logger.info(
                "ultrasound milestone: step=%s surface contact/down reached "
                "(ee=%s z=%.3f contact_z_max=%.3f alignment=%.3f)",
                step,
                _format_tensor(metrics.ee_pos),
                float(metrics.ee_pos[2].item()),
                _CONTACT_Z_MAX_M,
                metrics.alignment,
            )

        if (
            self._contact_pos is not None
            and self._twist_pos is None
            and metrics.alignment >= _TWIST_ALIGNMENT_THRESHOLD
        ):
            self._twist_pos = metrics.ee_pos.clone()
            logger.info(
                "ultrasound milestone: step=%s probe twist/alignment complete " "(alignment=%.3f threshold=%.3f ee=%s)",
                step,
                metrics.alignment,
                _TWIST_ALIGNMENT_THRESHOLD,
                _format_tensor(metrics.ee_pos),
            )

        if self._twist_pos is not None:
            swipe_distance = float(torch.linalg.norm(metrics.ee_pos[:2] - self._twist_pos[:2]).item())
            self._max_swipe_distance_m = max(self._max_swipe_distance_m, swipe_distance)
            if swipe_distance >= _SWIPE_START_DISTANCE_M:
                self._mark(
                    "swipe_started",
                    "ultrasound milestone: step=%s liver swipe started "
                    "(swipe_distance_m=%.3f threshold=%.3f ee=%s twist_start=%s)",
                    step,
                    swipe_distance,
                    _SWIPE_START_DISTANCE_M,
                    _format_tensor(metrics.ee_pos),
                    _format_tensor(self._twist_pos),
                )

    def _log_trajectory(self, ctx, metrics: "_SuccessMetrics") -> None:
        step = _policy_step(ctx)
        if step <= 1 or step % _TRAJECTORY_LOG_PERIOD_STEPS == 0:
            logger.info(
                "ultrasound trajectory: step=%s phase=%s target=%s ee=%s distance_m=%.3f alignment=%.3f "
                "swipe_distance_m=%.3f action=%s action_norm=%.4f action_max_abs=%.4f saturated_steps=%s",
                step,
                self._phase(),
                _format_tensor(metrics.target_pos),
                _format_tensor(metrics.ee_pos),
                metrics.distance_m,
                metrics.alignment,
                self._max_swipe_distance_m,
                _format_action(ctx),
                _policy_action_norm(ctx),
                _policy_action_max_abs(ctx),
                self._saturated_steps,
            )

    def _phase(self) -> str:
        if self._max_swipe_distance_m >= _SWIPE_START_DISTANCE_M:
            return "swipe"
        if self._twist_pos is not None:
            return "aligned"
        if self._contact_pos is not None:
            return "contact"
        return "approach"

    def _scan_goal_satisfied(self, metrics: "_SuccessMetrics", step: int) -> bool:
        return (
            step >= self._min_scan_steps
            and self._contact_pos is not None
            and self._twist_pos is not None
            and metrics.alignment >= self._alignment_threshold
            and self._max_swipe_distance_m >= _SCAN_COMPLETE_DISTANCE_M
        )

    def _success_state(self, ctx) -> dict:
        metrics = _success_metrics(ctx.env)
        near_target = metrics.distance_m <= self._target_tolerance_m and metrics.alignment >= self._alignment_threshold
        scan_complete = self._scan_goal_satisfied(metrics, _policy_step(ctx))
        return {
            "target_pos": metrics.target_pos,
            "ee_pos": metrics.ee_pos,
            "distance_m": metrics.distance_m,
            "alignment": metrics.alignment,
            "contact_seen": self._contact_pos is not None,
            "twist_seen": self._twist_pos is not None,
            "scan_distance_m": self._max_swipe_distance_m,
            "near_target": near_target,
            "scan_complete": scan_complete,
            "success": near_target or scan_complete,
            "target_tolerance_m": self._target_tolerance_m,
            "alignment_threshold": self._alignment_threshold,
            "scan_distance_threshold_m": _SCAN_COMPLETE_DISTANCE_M,
        }

    def _mark(self, key: str, message: str, *args) -> None:
        if key in self._seen:
            return
        self._seen.add(key)
        logger.info(message, *args)


def publish_obs(ctx) -> None:
    scene = ctx.env.unwrapped.scene
    room = scene["room_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
    wrist = scene["wrist_camera"].data.output["rgb"][0, ..., :3].cpu().numpy()
    joints = scene["robot"].data.joint_pos[0, :FRANKA_STATE_DIM].cpu().numpy()
    ctx.io.publish_camera("room", room)
    ctx.io.publish_camera("wrist", wrist)
    ctx.io.publish_state(joints)


@torch.no_grad()
def reset_to_home(env) -> None:
    """Reset env, then pre-roll the arm to the SETUP pose used in training.

    Mirrors the 40-step reset loop in workflows/robotic_ultrasound
    sim_with_dds.py: target absolute EE pose (SETUP, DOWN) and step the
    relative-mode IK controller until the arm settles there.
    """
    logger.info("resetting env to home")
    env.reset()

    device = env.unwrapped.device
    num_envs = env.unwrapped.num_envs
    setup_pos = torch.tensor(_SETUP_POS, device=device, dtype=torch.float32)
    down_quat = _quat_from_euler_xyz_deg(_DOWN_EULER_DEG, device=device)
    target_pose = torch.cat([setup_pos, down_quat], dim=-1).repeat(num_envs, 1)

    for _ in range(_RESET_STEPS):
        rel_action = _compute_relative_action(target_pose, _get_ee_pose(env))
        env.step(rel_action)

    _reset_success_state(env)
    logger.info("env reset to home")


def _policy_action(ctx) -> torch.Tensor | None:
    target = ctx.io.pop_action()
    if target is None:
        return None
    return torch.tensor(np.asarray(target, dtype=np.float32).flatten(), device=ctx.device)


class _SuccessMetrics:
    def __init__(self, *, target_pos: torch.Tensor, ee_pos: torch.Tensor, distance_m: float, alignment: float) -> None:
        self.target_pos = target_pos
        self.ee_pos = ee_pos
        self.distance_m = distance_m
        self.alignment = alignment


def _success_metrics(env) -> _SuccessMetrics:
    from isaaclab.utils.math import matrix_from_quat

    scene = env.unwrapped.scene
    target_pos = scene["goal_frame"].data.target_pos_w[0, 0].detach().cpu()
    ee_pos = scene["ee_frame"].data.target_pos_w[0, 0].detach().cpu()
    distance_m = float(torch.linalg.norm(target_pos - ee_pos).item())

    ee_rot = matrix_from_quat(scene["ee_frame"].data.target_quat_w[..., 0, :])
    goal_rot = matrix_from_quat(scene["goal_frame"].data.target_quat_w[..., 0, :])
    align_z = torch.sum(ee_rot[..., 2] * goal_rot[..., 2], dim=-1)
    align_x = torch.sum(ee_rot[..., 0] * goal_rot[..., 0], dim=-1)
    alignment = float(torch.minimum(align_z, align_x)[0].item())

    return _SuccessMetrics(target_pos=target_pos, ee_pos=ee_pos, distance_m=distance_m, alignment=alignment)


def _ultrasound_success_state(env) -> dict:
    last = getattr(env.unwrapped, "_ultrasound_success_last", None)
    if last is not None:
        env_index = 0
        return {
            "target_pos": last["target_pos"][env_index].detach().cpu(),
            "ee_pos": last["ee_pos"][env_index].detach().cpu(),
            "distance_m": float(last["distance"][env_index].item()),
            "alignment": float(last["alignment"][env_index].item()),
            "contact_seen": bool(last["contact_seen"][env_index].item()),
            "twist_seen": bool(last["twist_seen"][env_index].item()),
            "scan_distance_m": float(last["scan_distance"][env_index].item()),
            "near_target": bool(last["near_target"][env_index].item()),
            "scan_complete": bool(last["scan_complete"][env_index].item()),
            "success": bool(last["success"][env_index].item()),
            "target_tolerance_m": float(last["target_tolerance_m"]),
            "alignment_threshold": float(last["alignment_threshold"]),
            "scan_distance_threshold_m": float(last["scan_distance_threshold_m"]),
        }

    metrics = _success_metrics(env)
    return {
        "target_pos": metrics.target_pos,
        "ee_pos": metrics.ee_pos,
        "distance_m": metrics.distance_m,
        "alignment": metrics.alignment,
        "contact_seen": False,
        "twist_seen": False,
        "scan_distance_m": 0.0,
        "near_target": False,
        "scan_complete": False,
        "success": False,
        "target_tolerance_m": DEFAULT_TARGET_TOLERANCE_M,
        "alignment_threshold": DEFAULT_ALIGNMENT_THRESHOLD,
        "scan_distance_threshold_m": _SCAN_COMPLETE_DISTANCE_M,
    }


def _success_terminated(ctx) -> bool:
    terminated = getattr(ctx, "env_terminated", None)
    if terminated is None:
        return False
    return bool(terminated.any().item())


def _reset_success_state(env) -> None:
    from arena.tasks.ultrasound_liver_scan import reset_ultrasound_success_state

    unwrapped = env.unwrapped
    env_ids = torch.arange(unwrapped.num_envs, device=unwrapped.device)
    reset_ultrasound_success_state(unwrapped, env_ids)


def _get_ee_pose(env) -> torch.Tensor:
    """Current EE pose (pos + quat) in env-local frame. Shape (num_envs, 7)."""
    ee_data = env.unwrapped.scene["ee_frame"].data
    pos = ee_data.target_pos_w[:, 0, :] - env.unwrapped.scene.env_origins
    quat = ee_data.target_quat_w[:, 0, :]
    return torch.cat([pos, quat], dim=-1)


def _format_tensor(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values.detach().cpu().flatten().tolist()) + "]"


def _format_action(ctx) -> str:
    action = getattr(ctx, "policy_action_values", None)
    if action is None:
        return "[]"
    return _format_tensor(action)


def _policy_step(ctx) -> int:
    return int(getattr(ctx, "policy_step", 0))


def _policy_action_norm(ctx) -> float:
    return float(getattr(ctx, "policy_action_norm", 0.0))


def _policy_action_max_abs(ctx) -> float:
    return float(getattr(ctx, "policy_action_max_abs", 0.0))


def _compute_relative_action(target_pose: torch.Tensor, current_pose: torch.Tensor) -> torch.Tensor:
    """Convert absolute target → relative-pose action consumed by the IK controller."""
    from isaaclab.utils.math import compute_pose_error

    delta_pos, delta_angle = compute_pose_error(
        current_pose[:, :3], current_pose[:, 3:], target_pose[:, :3], target_pose[:, 3:], rot_error_type="axis_angle"
    )
    return torch.cat([delta_pos, delta_angle], dim=-1)


def _quat_from_euler_xyz_deg(euler_deg: tuple[float, float, float], device) -> torch.Tensor:
    from isaaclab.utils.math import quat_from_euler_xyz

    rx, ry, rz = (math.radians(a) for a in euler_deg)
    return quat_from_euler_xyz(
        torch.tensor(rx, device=device), torch.tensor(ry, device=device), torch.tensor(rz, device=device)
    )

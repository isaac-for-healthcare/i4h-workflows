# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 Assemble Trocar Zenoh runtime (Dex3 hand variant)."""

from __future__ import annotations

import time
import uuid

import numpy as np
import torch
from arena.runtimes.core.base import PolicyIO, logger
from common.config import get_robot_config, get_zenoh_config
from tqdm import trange

_ACTION_WAIT_TIMEOUT_S = 30.0
_G1_CONFIG = get_robot_config("g1")
_G1_BODY_JOINT_COUNT = _G1_CONFIG.body_joint_count
_G1_HAND_JOINT_COUNT = _G1_CONFIG.hand_joint_count


class AssembleTrocarPolicyIO(PolicyIO):
    def __init__(self, *, env_id: str, max_execution_steps: int | None = None) -> None:
        zenoh = get_zenoh_config(env_id)
        super().__init__(
            camera_keys=zenoh.camera_keys,
            state_key=zenoh.robot_state_key,
            command_key=zenoh.robot_command_key,
            action_dim=0,  # variable; subclass handles its own decode
            max_execution_steps=max_execution_steps,
        )

    def publish_observation(self, observation: dict) -> None:
        camera_obs = observation.get("camera_obs", {})
        for cam_key, obs_key in (
            ("front", "front_camera_rgb"),
            ("left_wrist", "left_wrist_camera_rgb"),
            ("right_wrist", "right_wrist_camera_rgb"),
        ):
            self.publish_camera(cam_key, _first_env_rgb(camera_obs[obs_key]))
        joints = _joint_positions(observation)
        self.publish_state(joints)

    def _on_command(self, cmd) -> None:  # type: ignore[override]
        if not self._command_matches_context(cmd):
            logger.debug(
                "ignoring stale assemble command: cmd_run=%s/%s/%s current=%s/%s/%s",
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
        if self._max_execution_steps is not None:
            actions = actions[: self._max_execution_steps]
        with self._cmd_lock:
            if self._action_queue:
                return
            for row in actions:
                self._action_queue.append(row.copy())


def run(ctx, args_cli) -> None:
    if not _G1_CONFIG.has_dex_hand:
        raise RuntimeError(
            f"Active humanoid '{_G1_CONFIG.id}' has no dex hand; the assemble_trocar task "
            "requires a Dex3-equipped robot. Use the G1 robot profile or "
            "pick a different env."
        )

    max_timesteps = args_cli.max_timesteps or 500
    save_episode_cb = getattr(ctx, "save_episode_cb", None)
    save_successful_episode_cb = getattr(ctx, "save_successful_episode_cb", None) or save_episode_cb
    discard_episode_cb = getattr(ctx, "discard_episode_cb", None)
    save_all = bool(getattr(ctx, "save_all_episodes", False))
    if args_cli.max_attempts and args_cli.max_attempts > 0:
        max_attempts_per_episode = args_cli.max_attempts
    elif save_all or save_episode_cb is None:
        max_attempts_per_episode = 1
    else:
        max_attempts_per_episode = 3
    progress_label = "saved" if save_episode_cb is not None else "completed"
    policy_job_id = uuid.uuid4().hex[:8]
    logger.info(
        "Assemble Trocar policy job: episodes=%s max_timesteps=%s save_all=%s max_attempts_per_episode=%s",
        args_cli.episodes,
        max_timesteps,
        save_all,
        max_attempts_per_episode,
    )
    # Original Assemble Trocar eval is closed-loop: only use the first action
    # from each GR00T chunk, then re-query the policy on the next sim step.
    with AssembleTrocarPolicyIO(env_id=ctx.env_id, max_execution_steps=1) as io:
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
                    "Assemble Trocar policy episode %s/%s exhausted %s attempts; marking failed and moving to next requested episode",
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
                "Assemble Trocar policy episode %s/%s attempt %s/%s started; %s=%s/%s",
                current_episode,
                args_cli.episodes,
                episode_attempts,
                max_attempts_per_episode,
                progress_label,
                saved,
                args_cli.episodes,
            )
            status = _run_episode(ctx, io, obs, max_timesteps=max_timesteps)
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
                "Assemble Trocar policy episode %s/%s attempt %s/%s: %s",
                completed_episode,
                args_cli.episodes,
                completed_attempt,
                max_attempts_per_episode,
                status,
            )
        logger.info(
            "Assemble Trocar policy job complete: %s=%s/%s failed=%s total_attempts=%s",
            progress_label,
            saved,
            args_cli.episodes,
            failed,
            total_attempts,
        )


@torch.no_grad()
def _run_episode(ctx, io, obs: dict, *, max_timesteps: int) -> str:
    for step in trange(max_timesteps, desc=ctx.env_id, leave=False):
        if not _ready(ctx):
            return "aborted"
        io.publish_observation(obs)
        action = _wait_for_action(ctx, io)
        if action is None:
            return "timeout"
        action_tensor = torch.as_tensor(action, device=ctx.env.device, dtype=torch.float32)
        if action_tensor.ndim == 1:
            action_tensor = action_tensor.unsqueeze(0)
        if action_tensor.shape[0] == 1 and ctx.env.num_envs > 1:
            action_tensor = action_tensor.repeat(ctx.env.num_envs, 1)
        obs, _, terminated, truncated, _ = ctx.env.step(action_tensor)
        if bool(terminated.any().item()) or _task_stage(ctx.env) >= 5:
            _log_assemble_state(ctx.env, step + 1, "success")
            return "completed"
        if bool(truncated.any().item()):
            if _task_stage(ctx.env) >= 5:
                _log_assemble_state(ctx.env, step + 1, "success")
                return "completed"
            _log_assemble_state(ctx.env, step + 1, "terminated_without_success")
            return "timeout"
    if _task_stage(ctx.env) >= 5:
        _log_assemble_state(ctx.env, max_timesteps, "success")
        return "completed"
    _log_assemble_state(ctx.env, max_timesteps, "timeout")
    return "timeout"


def _wait_for_action(ctx, io) -> np.ndarray | None:
    wait_started = time.monotonic()
    while _ready(ctx):
        action = io.pop_action()
        if action is not None:
            return action
        if time.monotonic() - wait_started > _ACTION_WAIT_TIMEOUT_S:
            logger.warning(
                "timed out waiting for current-run policy action; restart the policy daemon if stale commands were rejected"
            )
            io.clear_actions()
            return None
        time.sleep(0.01)
    return None


def _ready(ctx) -> bool:
    return ctx.simulation_app.is_running() and not ctx.controller.should_abort()


def _task_stage(env) -> int:
    try:
        from arena.tasks.assemble_trocar import get_task_stage

        return int(get_task_stage(env)[0].item())
    except Exception:
        return -1


def _log_assemble_state(env, step: int, status: str) -> None:
    state = _assemble_state(env, status=status)
    thresholds = _assemble_thresholds()
    logger.info(
        "Assemble Trocar %s state: step=%s stage=%s trocar_1=%s trocar_2=%s "
        "center_distance_m=%.4f center_threshold_m=%.4f tip_distance_m=%.4f tip_threshold_m=%.4f "
        "axis_angle_rad=%.4f angle_threshold_rad=%.4f lift_z_min_m=%.4f "
        "placement_bounds=x[%.3f,%.3f] y[%.3f,%.3f] z<%.3f",
        status,
        step,
        state["stage"],
        _format_vec3(state["trocar_1"]),
        _format_vec3(state["trocar_2"]),
        state["center_distance_m"],
        thresholds["insertion_dist_threshold"],
        state["tip_distance_m"],
        thresholds["tip_align_threshold"],
        state["angle_rad"],
        thresholds["insertion_angle_threshold"],
        thresholds["table_height"] + thresholds["lift_threshold"],
        min(thresholds["placement_x_min"], thresholds["placement_x_max"]),
        max(thresholds["placement_x_min"], thresholds["placement_x_max"]),
        min(thresholds["placement_y_min"], thresholds["placement_y_max"]),
        max(thresholds["placement_y_min"], thresholds["placement_y_max"]),
        thresholds["placement_z_min"],
    )


def _assemble_state(env, *, status: str) -> dict:
    cached = getattr(env.unwrapped, "_assemble_trocar_last", None)
    if cached is not None and status != "timeout":
        cached_status = str(cached.get("status", ""))
        if status == "success" and cached_status == "success":
            return _cached_assemble_state(cached)
        if status == "terminated_without_success" and cached_status == "object_drop":
            return _cached_assemble_state(cached)

    scene = env.unwrapped.scene
    pos1 = scene["trocar_1"].data.root_pos_w[0].detach().cpu()
    pos2 = scene["trocar_2"].data.root_pos_w[0].detach().cpu()
    return {
        "stage": _task_stage(env),
        "trocar_1": pos1,
        "trocar_2": pos2,
        "center_distance_m": float(torch.linalg.norm(pos1 - pos2).item()),
        "tip_distance_m": _trocar_tip_distance(env),
        "angle_rad": _trocar_axis_angle(env),
    }


def _cached_assemble_state(cached: dict) -> dict:
    env_index = 0
    return {
        "stage": int(cached["stage"][env_index].item()),
        "trocar_1": cached["trocar_1"][env_index].detach().cpu(),
        "trocar_2": cached["trocar_2"][env_index].detach().cpu(),
        "center_distance_m": float(cached["center_distance"][env_index].item()),
        "tip_distance_m": float(cached["tip_distance"][env_index].item()),
        "angle_rad": float(cached["axis_angle"][env_index].item()),
    }


def _trocar_axis_angle(env) -> float:
    try:
        from isaaclab.utils.math import matrix_from_quat

        scene = env.unwrapped.scene
        rot1 = matrix_from_quat(scene["trocar_1"].data.root_quat_w[0:1])
        rot2 = matrix_from_quat(scene["trocar_2"].data.root_quat_w[0:1])
        axis1 = rot1[0, :, 2]
        axis2 = rot2[0, :, 2]
        cosine = torch.clamp(torch.dot(axis1, axis2), -1.0, 1.0)
        return float(torch.acos(cosine).item())
    except Exception:
        return float("nan")


def _trocar_tip_distance(env) -> float:
    try:
        from types import SimpleNamespace

        from arena.tasks.assemble_trocar import _trocar_tip_position

        unwrapped = env.unwrapped
        tip1 = _trocar_tip_position(unwrapped, SimpleNamespace(name="trocar_1"))[0].detach().cpu()
        tip2 = _trocar_tip_position(unwrapped, SimpleNamespace(name="trocar_2"))[0].detach().cpu()
        return float(torch.linalg.norm(tip1 - tip2).item())
    except Exception:
        return float("nan")


def _assemble_thresholds() -> dict:
    try:
        from arena.tasks.assemble_trocar import _stage_params

        params = _stage_params()
        return {
            "table_height": float(params["table_height"]),
            "lift_threshold": float(params["lift_threshold"]),
            "tip_align_threshold": float(params["tip_align_threshold"]),
            "insertion_dist_threshold": float(params["insertion_dist_threshold"]),
            "insertion_angle_threshold": float(params["insertion_angle_threshold"]),
            "placement_x_min": float(params["placement_x_min"]),
            "placement_x_max": float(params["placement_x_max"]),
            "placement_y_min": float(params["placement_y_min"]),
            "placement_y_max": float(params["placement_y_max"]),
            "placement_z_min": float(params["placement_z_min"]),
        }
    except Exception:
        return {
            "table_height": 0.85483,
            "lift_threshold": 0.15,
            "tip_align_threshold": 0.015,
            "insertion_dist_threshold": 0.05,
            "insertion_angle_threshold": 0.15,
            "placement_x_min": -1.8,
            "placement_x_max": -1.4,
            "placement_y_min": 1.5,
            "placement_y_max": 1.8,
            "placement_z_min": 0.9,
        }


def _format_vec3(values: torch.Tensor) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values.detach().cpu().flatten()[:3].tolist()) + "]"


def _joint_positions(observation: dict) -> np.ndarray:
    policy_obs = observation.get("policy", {})
    if "robot_joint_state" not in policy_obs or "robot_dex3_joint_state" not in policy_obs:
        raise KeyError(
            f"could not find Assemble Trocar state keys in policy observation keys: {list(policy_obs.keys())}"
        )
    body = _to_numpy(policy_obs["robot_joint_state"])
    dex3 = _to_numpy(policy_obs["robot_dex3_joint_state"])
    if body.ndim > 1:
        body = body[0]
    if dex3.ndim > 1:
        dex3 = dex3[0]
    return np.concatenate([body[:_G1_BODY_JOINT_COUNT], dex3[:_G1_HAND_JOINT_COUNT]], axis=0).astype(
        np.float32, copy=False
    )


def _first_env_rgb(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    array = np.asarray(value)
    if array.ndim == 4:
        array = array[0]
    return array[..., :3].astype(np.uint8, copy=False)


def _to_numpy(value) -> np.ndarray:
    if hasattr(value, "detach"):
        value = value.detach().cpu().numpy()
    return np.asarray(value, dtype=np.float32)

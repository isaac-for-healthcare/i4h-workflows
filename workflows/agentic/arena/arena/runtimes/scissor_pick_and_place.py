# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM scissor pick-and-place Zenoh runtime."""

from __future__ import annotations

import numpy as np
import torch
from arena.arena_config import get_arena_config
from arena.runtimes.core.base import PolicyIO, logger, ready, run_policy_episode  # noqa: F401 (ready re-exported)
from arena.tasks.scissor_pick_and_place import success_scissors_in_tray, success_scissors_placed
from common.config import get_robot_config, get_zenoh_config
from common.joint_utils import isaaclab_rad_to_lerobot, lerobot_to_isaaclab_rad

_SCISSORS_SETTLE_MOVE_M = 0.004
_SO101_CONFIG = get_robot_config("so101")
_ROBOT_ACTION_DIM = _SO101_CONFIG.action_dim
_ROBOT_ISAACLAB_JOINT_POS_LIMIT_RANGE = _SO101_CONFIG.isaaclab_joint_pos_limit_range
_ROBOT_JOINT_NAMES = _SO101_CONFIG.joint_names
_ROBOT_LEROBOT_JOINT_POS_LIMIT_RANGE = _SO101_CONFIG.lerobot_joint_pos_limit_range


class ScissorPolicyIO(PolicyIO):
    def __init__(self, *, env_id: str) -> None:
        zenoh = get_zenoh_config(env_id)
        super().__init__(
            camera_keys=zenoh.camera_keys,
            state_key=zenoh.robot_state_key,
            command_key=zenoh.robot_command_key,
            action_dim=_ROBOT_ACTION_DIM,
        )


def run_policy_based_episode(
    ctx,
    *,
    max_timesteps: int,
    external_success: bool = False,
) -> str:
    """Run one policy episode.

    When ``external_success=True`` (filter-success recording mode), the env's
    success termination is forced off by ``setup_recording``; we detect success
    here via :func:`success_scissors_placed` and tell the loop to stop. When
    False, the env's own termination drives episode end.
    """
    logger.info(
        "scissor success target: scissors settle in tray and arm returns home " "(move_tolerance_m=%.3f)",
        _SCISSORS_SETTLE_MOVE_M,
    )
    status = run_policy_episode(
        ctx,
        max_timesteps=max_timesteps,
        publish_obs=publish_obs,
        policy_action=_policy_action,
        success_condition=_scissor_success if external_success else None,
        stop_on_env_done=not external_success,
    )
    if status == "timeout":
        if _scissors_in_tray(ctx.env):
            logger.info("scissor timeout accepted as success because scissors are in tray")
            _log_success(ctx)
            return "completed"
        _log_timeout(ctx)
    elif status == "completed":
        _log_success(ctx)
    return status


def _scissor_success(ctx) -> bool:
    return bool(success_scissors_placed(ctx.env).any().item())


def publish_obs(ctx) -> None:
    scene = ctx.env.unwrapped.scene
    room = scene["room"].data.output["rgb"][0, ..., :3].cpu().numpy()
    wrist = scene["wrist"].data.output["rgb"][0, ..., :3].cpu().numpy()
    robot = scene["robot"]
    full = robot.data.joint_pos[0:1].cpu().numpy()
    if full.shape[-1] != len(_ROBOT_JOINT_NAMES):
        full = full[:, _active_joint_indices(robot)]
    joints_lerobot = isaaclab_rad_to_lerobot(
        full, _ROBOT_ISAACLAB_JOINT_POS_LIMIT_RANGE, _ROBOT_LEROBOT_JOINT_POS_LIMIT_RANGE
    ).flatten()

    ctx.io.publish_camera("room", room)
    ctx.io.publish_camera("wrist", wrist)
    ctx.io.publish_state(joints_lerobot)


@torch.no_grad()
def sync_robot_joints(env, joint_positions: torch.Tensor) -> None:
    robot = env.unwrapped.scene["robot"]
    target = joint_positions.detach().to(device=env.device, dtype=robot.data.joint_pos.dtype)
    if target.ndim == 1:
        target = target.unsqueeze(0)
    if target.shape[0] == 1 and env.unwrapped.num_envs > 1:
        target = target.repeat(env.unwrapped.num_envs, 1)
    if target.shape[-1] != _ROBOT_ACTION_DIM:
        raise ValueError(f"expected {_ROBOT_ACTION_DIM} joint positions, got {target.shape[-1]}")

    joint_ids = [_joint_index(robot, name, index) for index, name in enumerate(_ROBOT_JOINT_NAMES)]
    zeros = torch.zeros_like(target)
    robot.write_joint_state_to_sim(target, zeros, joint_ids=joint_ids)
    robot.set_joint_position_target(target, joint_ids=joint_ids)
    robot.set_joint_velocity_target(zeros, joint_ids=joint_ids)
    robot.write_data_to_sim()


def _policy_action(ctx) -> torch.Tensor | None:
    target = ctx.io.pop_action()
    if target is None:
        return None
    target_rad = lerobot_to_isaaclab_rad(
        target.reshape(1, _ROBOT_ACTION_DIM),
        _ROBOT_LEROBOT_JOINT_POS_LIMIT_RANGE,
        _ROBOT_ISAACLAB_JOINT_POS_LIMIT_RANGE,
    )
    home = get_arena_config(ctx.env_id).home_joint_pos_rad
    if home is None:
        raise ValueError(f"arena config for {ctx.env_id!r} is missing home_joint_pos_rad")
    target_rad = target_rad - np.asarray(home, dtype=np.float32).reshape(1, _ROBOT_ACTION_DIM)
    return torch.tensor(target_rad.flatten().astype(np.float32), device=ctx.device)


def _scissors_in_tray(env) -> bool:
    in_tray = success_scissors_in_tray(env)
    return bool(in_tray.any().item())


def _log_timeout(ctx) -> None:
    state = _scissor_success_state(ctx.env)
    logger.info(
        "scissor success timeout state: step=%s in_tray=%s settled=%s arm_home=%s "
        "scissors_position=%s ee_position=%s ee_home=%s ee_home_error_m=%.3f tolerance_m=%.3f",
        _policy_step(ctx),
        state["in_tray"],
        state["settled"],
        state["arm_home"],
        _format_tensor(state["scissors_pos"]),
        _format_tensor(state["ee_pos"]),
        _format_tensor(state["home_ee_pos"]),
        state["ee_home_error_m"],
        state["home_tolerance_m"],
    )


def _log_success(ctx) -> None:
    state = _scissor_success_state(ctx.env)
    logger.info(
        "scissor success final state: step=%s in_tray=%s settled=%s arm_home=%s "
        "scissors_position=%s ee_position=%s ee_home=%s ee_home_error_m=%.3f tolerance_m=%.3f",
        _policy_step(ctx),
        state["in_tray"],
        state["settled"],
        state["arm_home"],
        _format_tensor(state["scissors_pos"]),
        _format_tensor(state["ee_pos"]),
        _format_tensor(state["home_ee_pos"]),
        state["ee_home_error_m"],
        state["home_tolerance_m"],
    )


def _scissor_success_state(env) -> dict:
    last = getattr(env.unwrapped, "_scissor_success_last", None)
    if last is not None:
        env_index = 0
        return {
            "in_tray": bool(last["in_tray"][env_index].item()),
            "settled": bool(last["settled"][env_index].item()),
            "arm_home": bool(last["arm_home"][env_index].item()),
            "scissors_pos": last["scissors_pos"][env_index].detach().cpu(),
            "ee_pos": last["ee_pos"][env_index].detach().cpu(),
            "home_ee_pos": last["home_ee_pos"][env_index].detach().cpu(),
            "ee_home_error_m": float(last["ee_home_error_m"][env_index].item()),
            "home_tolerance_m": float(last["home_tolerance_m"]),
        }

    return {
        "in_tray": _scissors_in_tray(env),
        "settled": False,
        "arm_home": False,
        "scissors_pos": _scissors_world_position(env).detach().cpu(),
        "ee_pos": torch.full((3,), float("nan")),
        "home_ee_pos": torch.full((3,), float("nan")),
        "ee_home_error_m": float("nan"),
        "home_tolerance_m": float("nan"),
    }


def _scissors_world_position(env) -> torch.Tensor:
    return env.unwrapped.scene["scissors"].data.root_pos_w[0]


def _format_tensor(values: torch.Tensor) -> str:
    return _format_values(float(value) for value in values.detach().cpu().flatten().tolist())


def _format_values(values) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in values) + "]"


def _policy_step(ctx) -> int:
    return int(getattr(ctx, "policy_step", 0))


def _active_joint_indices(robot) -> list[int]:
    return [_joint_index(robot, name, idx) for idx, name in enumerate(_ROBOT_JOINT_NAMES)]


def _joint_index(robot, joint_name: str, fallback: int) -> int:
    if hasattr(robot, "find_joints"):
        found = robot.find_joints(joint_name, preserve_order=True)
        for value in (found[0], found[1]):
            if hasattr(value, "numel") and value.numel() > 0:
                return int(value[0])
            if len(value) > 0 and isinstance(value[0], int):
                return int(value[0])
    return fallback

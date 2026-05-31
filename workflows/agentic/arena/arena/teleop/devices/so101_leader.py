# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO-ARM 101 kinematic leader teleop.

Wraps `leisaac.devices.SO101Leader` and converts leader joint values into
sim-frame joint commands. Joint-name lookups (`wrist_flex`, `gripper`)
assume the active robot is SO-ARM 101; this module must only be
imported from the `so101_leader` branch of `arena.teleop`, which is
only reachable when `ROBOT_TELEOP_DEVICES` includes `"so101_leader"`.
"""

from __future__ import annotations

import sys
from pathlib import Path

import torch
from common.config import get_robot_config

LEADER_CALIBRATION_PATH = (
    Path.home() / ".cache/huggingface/lerobot/calibration/teleoperators/so101_leader/so101_leader.json"
)
_SO101_CONFIG = get_robot_config("so101")
_SO101_LEADER_CONFIG = _SO101_CONFIG.teleop_device_configs.get("so101_leader") or {}


def make_interface(env, args_cli):
    if "wrist_flex" not in _SO101_CONFIG.joint_names or "gripper" not in _SO101_CONFIG.joint_names:
        raise RuntimeError("--teleop-device so101_leader requires the SO-ARM 101 robot profile.")
    from leisaac.devices import SO101Leader

    calibration_path = _leader_calibration_path()
    if calibration_path is None and not sys.stdin.isatty():
        raise RuntimeError(
            "SO101 leader calibration not found. Run in a foreground terminal or use --teleop-device keyboard."
        )
    kwargs = {"calibration_file_name": str(calibration_path)} if calibration_path is not None else {}
    leader = SO101Leader(
        env,
        port=args_cli.teleop_port,
        recalibrate=args_cli.teleop_recalibrate,
        **kwargs,
    )
    _use_direct_leader_joint_degrees(leader)
    leader.advance = lambda: _advance_direct_joint_radians(leader, env, _SO101_CONFIG.joint_names)
    return leader


def leader_joints_to_sim_joints(
    actions: torch.Tensor,
    leader_reference: torch.Tensor,
    home_joint_pos_rad: list[float] | tuple[float, ...],
) -> torch.Tensor:
    joint_names = _SO101_CONFIG.joint_names
    sim_signs = _SO101_LEADER_CONFIG.get("sim_signs_by_joint") or {}
    wrist_flex_baseline_deg = float(_SO101_LEADER_CONFIG.get("wrist_flex_baseline_deg", 0.0))
    gripper_close_offset_deg = float(_SO101_LEADER_CONFIG.get("gripper_close_offset_deg", 0.0))

    wrist_flex_index = joint_names.index("wrist_flex")
    gripper_index = joint_names.index("gripper")

    offsets = torch.tensor(home_joint_pos_rad, device=actions.device, dtype=actions.dtype).reshape(1, -1)
    signs = torch.tensor(
        [sim_signs.get(joint_name, 1.0) for joint_name in joint_names],
        device=actions.device,
        dtype=actions.dtype,
    ).reshape(1, -1)
    mapped = offsets + actions * signs
    mapped[:, wrist_flex_index] = torch.deg2rad(
        torch.tensor(wrist_flex_baseline_deg, device=actions.device, dtype=actions.dtype)
    ) + (actions[:, wrist_flex_index] - leader_reference[:, wrist_flex_index])
    mapped[:, gripper_index] = actions[:, gripper_index] + torch.deg2rad(
        torch.tensor(gripper_close_offset_deg, device=actions.device, dtype=actions.dtype)
    )
    return mapped


def _use_direct_leader_joint_degrees(leader) -> None:
    from leisaac.devices.lerobot.common.motors import MotorNormMode

    for joint_name in _SO101_CONFIG.joint_names[:-1]:
        leader._bus.motors[joint_name].norm_mode = MotorNormMode.DEGREES
    leader._motor_limits = {
        joint_name: _SO101_CONFIG.isaaclab_joint_pos_limit_range[index]
        for index, joint_name in enumerate(_SO101_CONFIG.joint_names)
    }


def _advance_direct_joint_radians(leader, env, joint_names: list[str]):
    action = leader.input2action()
    if action is None:
        return env.action_manager.action
    if not action["started"]:
        return None
    if action["reset"]:
        return action
    values = [action["joint_state"][joint_name] for joint_name in joint_names]
    joints = torch.deg2rad(torch.tensor(values, device=env.device, dtype=torch.float32)).reshape(1, -1)
    if env.num_envs > 1:
        joints = joints.repeat(env.num_envs, 1)
    return joints


def _leader_calibration_path() -> Path | None:
    return LEADER_CALIBRATION_PATH if LEADER_CALIBRATION_PATH.exists() else None

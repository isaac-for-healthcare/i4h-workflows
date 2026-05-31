# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka ultrasound teleop adapter — SE3 keyboard / gamepad.

The ultrasound env's action manager expects a 6D relative-IK pose action:
``[dx, dy, dz, drx, dry, drz]`` where ``drx/dry/drz`` are axis-angle.
``Se3Keyboard`` / ``Se3Gamepad`` emit 6D delta poses in *euler* form; we
convert to axis-angle via the EE pose path used by
``workflows/robotic_ultrasound/scripts/simulation/environments/teleoperation/teleop_se3_agent.py``.
"""

from __future__ import annotations

import torch


def make_teleop_interface(env, args_cli):
    device_name = (args_cli.teleop_device or "").lower()
    if device_name == "keyboard":
        from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg

        device = Se3Keyboard(
            Se3KeyboardCfg(
                pos_sensitivity=0.05 * getattr(args_cli, "teleop_sensitivity", 1.0),
                rot_sensitivity=0.15 * getattr(args_cli, "teleop_sensitivity", 1.0),
            )
        )
        _display_keyboard_controls()
    elif device_name == "gamepad":
        from isaaclab.devices import Se3Gamepad, Se3GamepadCfg

        device = Se3Gamepad(
            Se3GamepadCfg(
                pos_sensitivity=0.05 * getattr(args_cli, "teleop_sensitivity", 1.0),
                rot_sensitivity=0.15 * getattr(args_cli, "teleop_sensitivity", 1.0),
            )
        )
        _display_gamepad_controls()
    else:
        raise ValueError(f"unsupported teleop device for Franka: {args_cli.teleop_device!r}")

    return _Se3TeleopAdapter(env, device)


def _print_header(title: str):
    def print_command(control: str, info: str) -> None:
        control += " " * (20 - len(control))
        print(f"{control}\t{info}")

    print("")
    print("=" * 80)
    print(title)
    print("=" * 80)
    print_command("B", "start control")
    print_command("N", "mark episode successful / save")
    print_command("R", "reset simulation / discard attempt")
    print("-" * 80)
    return print_command


def _display_keyboard_controls() -> None:
    print_command = _print_header("Franka Keyboard Teleop Controls")
    print_command("W / S", "end-effector +X / -X")
    print_command("A / D", "end-effector +Y / -Y")
    print_command("Q / E", "end-effector +Z / -Z")
    print_command("Z / X", "roll + / -")
    print_command("T / G", "pitch + / -")
    print_command("C / V", "yaw + / -")
    print("-" * 80)
    print("Action: relative end-effector pose [dx, dy, dz, droll, dpitch, dyaw].")
    print("=" * 80)
    print("")


def _display_gamepad_controls() -> None:
    print_command = _print_header("Franka Gamepad Teleop Controls")
    print_command("sticks/triggers", "move end-effector with IsaacLab Se3Gamepad bindings")
    print("-" * 80)
    print("Action: relative end-effector pose [dx, dy, dz, droll, dpitch, dyaw].")
    print("=" * 80)
    print("")


class _Se3TeleopAdapter:
    """Wraps Se3Keyboard/Se3Gamepad so ``advance()`` returns the IK action."""

    def __init__(self, env, device):
        self._env = env
        self._device = device

    def reset(self) -> None:
        self._device.reset()

    def add_callback(self, key: str, func) -> None:
        adder = getattr(self._device, "add_callback", None)
        if adder is not None:
            adder(key, func)

    @torch.no_grad()
    def advance(self) -> torch.Tensor | None:
        from isaaclab.managers import SceneEntityCfg
        from isaaclab.utils import math as math_utils

        delta_pose = self._device.advance()
        if delta_pose is None:
            return None
        env = self._env.unwrapped
        delta_pose = delta_pose.to(env.device).repeat(env.num_envs, 1)
        delta_pos = delta_pose[:, :3]
        delta_rot = math_utils.quat_from_euler_xyz(delta_pose[:, 3], delta_pose[:, 4], delta_pose[:, 5])

        robot_cfg = SceneEntityCfg("robot", joint_names=["panda_joint.*"], body_names=["TCP"])
        robot_cfg.resolve(env.scene)
        ee_pose_w = env.scene["robot"].data.body_state_w[:, robot_cfg.body_ids[0], 0:7].clone()
        ee_pos_w, ee_rot_w = ee_pose_w[:, :3], ee_pose_w[:, 3:7]

        target_pos_w, target_rot_w = math_utils.combine_frame_transforms(ee_pos_w, ee_rot_w, delta_pos, delta_rot)
        delta_pos, delta_rot = math_utils.compute_pose_error(
            ee_pos_w, ee_rot_w, target_pos_w, target_rot_w, rot_error_type="quat"
        )
        delta_rot[delta_rot.abs() < 1e-6] = 0.0
        axis_angle = math_utils.axis_angle_from_quat(delta_rot)
        return torch.cat([delta_pos, axis_angle], dim=1)

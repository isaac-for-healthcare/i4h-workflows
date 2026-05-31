# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""SO-ARM 101 keyboard teleop device.

Wraps leisaac's `Se3Keyboard` for the SO-ARM 101 joint layout. This adapter is
intentionally SO-ARM-specific: it uses SO-ARM 101 joint constants and leisaac's
keyboard event handling, then returns joint-space actions without calling
`env.cfg.preprocess_device_action`.
"""

from __future__ import annotations

import numpy as np
import torch
from common.config import get_robot_config

_SO101_CONFIG = get_robot_config("so101")
_ROBOT_ACTION_DIM = _SO101_CONFIG.action_dim
_ROBOT_ARM_JOINT_COUNT = _SO101_CONFIG.arm_joint_count or _SO101_CONFIG.body_joint_count
_ROBOT_JOINT_NAMES = _SO101_CONFIG.joint_names

_POS_KEYS = ["Q", "W", "E", "A", "S", "D", "Z", "C"]
_NEG_KEYS = ["U", "I", "O", "J", "K", "L", "X", "M"]


def make_interface(env, args_cli):
    from leisaac.devices import Se3Keyboard

    joint_dim = _ROBOT_ARM_JOINT_COUNT + 1
    gripper_count = _ROBOT_ACTION_DIM - _ROBOT_ARM_JOINT_COUNT
    if joint_dim > len(_POS_KEYS):
        raise ValueError(
            f"SO-ARM 101 keyboard teleop supports up to {len(_POS_KEYS)} logical inputs, "
            f"got arm_joint_count + 1 = {joint_dim}"
        )

    device = Se3Keyboard(env, sensitivity=0.25 * args_cli.teleop_sensitivity)
    sensitivity = device.sensitivity

    # Resize delta buffer + reset hook to SO-ARM 101 joint_dim instead of leisaac's hardcoded 6.
    device._delta_pos = np.zeros(joint_dim)
    device.reset = lambda: setattr(device, "_delta_pos", np.zeros(joint_dim))

    mapping: dict[str, np.ndarray] = {}
    for axis in range(joint_dim):
        plus = np.zeros(joint_dim)
        plus[axis] = 1.0
        mapping[_POS_KEYS[axis]] = plus * sensitivity
        minus = np.zeros(joint_dim)
        minus[axis] = -1.0
        mapping[_NEG_KEYS[axis]] = minus * sensitivity
    device._INPUT_KEY_MAPPING = mapping
    _display_controls(joint_dim)

    def _advance() -> torch.Tensor | dict | None:
        action_dict = device.input2action()
        if action_dict is None:
            return env.action_manager.action
        if not action_dict["started"]:
            return None
        if action_dict["reset"]:
            return action_dict

        joint_state = action_dict["joint_state"]
        if isinstance(joint_state, np.ndarray):
            joint_state = torch.as_tensor(joint_state, device=env.device, dtype=torch.float32)
        else:
            joint_state = joint_state.to(device=env.device, dtype=torch.float32)

        out = torch.zeros(env.num_envs, _ROBOT_ACTION_DIM, device=env.device, dtype=torch.float32)
        out[:, :_ROBOT_ARM_JOINT_COUNT] = joint_state[:_ROBOT_ARM_JOINT_COUNT]
        gripper_value = joint_state[_ROBOT_ARM_JOINT_COUNT]
        for finger in range(gripper_count):
            out[:, _ROBOT_ARM_JOINT_COUNT + finger] = gripper_value
        return out

    device.advance = _advance
    return device


def _display_controls(joint_dim: int) -> None:
    def print_command(control: str, info: str) -> None:
        control += " " * (20 - len(control))
        print(f"{control}\t{info}")

    print("")
    print("=" * 80)
    print("SO-ARM 101 Keyboard Teleop Controls")
    print("=" * 80)
    print_command("B", "start control")
    print_command("N", "mark episode successful / save")
    print_command("R", "reset simulation / discard attempt")
    print("-" * 80)
    for axis in range(joint_dim):
        label = _ROBOT_JOINT_NAMES[axis] if axis < len(_ROBOT_JOINT_NAMES) else f"Joint {axis + 1}"
        print_command(f"{_POS_KEYS[axis]} / {_NEG_KEYS[axis]}", f"{label} +ve / -ve")
    print("=" * 80)
    print("")

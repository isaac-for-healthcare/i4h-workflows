# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""G1 humanoid teleop adapter — 23D keyboard device.

The locomanip teleop envs use a 23-dim WBC action:
``[left_grip, right_grip, left_wrist_pos(3), left_wrist_quat(4),
right_wrist_pos(3), right_wrist_quat(4), navigate_cmd(3),
base_height, torso_rpy(3)]``.

``keyboard_23d`` emits the action in this layout via
``KeyboardTo23DAdapter.advance()``.
"""

from __future__ import annotations

import torch


def make_teleop_interface(env, args_cli):
    device_name = (args_cli.teleop_device or "").lower()
    action_dim = int(env.unwrapped.action_space.shape[-1])
    if device_name == "keyboard_23d":
        if action_dim != 23:
            raise ValueError(
                f"keyboard_23d emits 23D WBC actions, but this env expects {action_dim}D actions. "
                "Use the g1_wbc_pink teleop embodiment for locomanip teleop; "
                "assemble_trocar needs Rheo's separate XR WBC teleop env ported."
            )
        from arena.teleop.devices.keyboard_23d import KeyboardTo23DAdapter, KeyboardTo23DConfig

        adapter = KeyboardTo23DAdapter(
            KeyboardTo23DConfig(
                pos_sensitivity=0.01 * getattr(args_cli, "teleop_sensitivity", 1.0),
                rot_sensitivity=0.05 * getattr(args_cli, "teleop_sensitivity", 1.0),
                default_base_height=getattr(args_cli, "teleop_base_height", 0.75),
            ),
            sim_device=str(env.unwrapped.device),
        )
        return _HumanoidAdapter(env, adapter, broadcast=True)

    raise ValueError(f"unsupported teleop device for humanoid: {args_cli.teleop_device!r}")


class _HumanoidAdapter:
    """Thin wrapper: forwards reset/add_callback; ``advance()`` shapes the action."""

    def __init__(self, env, device, *, broadcast: bool):
        self._env = env
        self._device = device
        # keyboard_23d returns a (23,) vector that must be broadcast across
        # num_envs.
        self._broadcast = broadcast

    def reset(self) -> None:
        reset_fn = getattr(self._device, "reset", None)
        if callable(reset_fn):
            reset_fn()

    def add_callback(self, key: str, func) -> None:
        adder = getattr(self._device, "add_callback", None)
        if adder is not None:
            adder(key, func)

    def advance(self) -> torch.Tensor | None:
        actions = self._device.advance()
        if actions is None:
            return None
        env = self._env.unwrapped
        actions = actions.to(env.device, dtype=torch.float32)
        if self._broadcast and actions.dim() == 1:
            actions = actions.unsqueeze(0).repeat(env.num_envs, 1)
        return actions

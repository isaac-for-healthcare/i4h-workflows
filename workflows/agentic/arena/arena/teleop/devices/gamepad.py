# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Game controller (gamepad) teleop device.

Wraps IsaacLab's :class:`isaaclab.devices.gamepad.Se3Gamepad` as an
``@register_device`` ``TeleopDeviceBase`` so the humanoid envs can look it up
via ``device_registry.get_device_by_name("gamepad")`` and pass the resulting
``DevicesCfg`` through to IsaacLab. Mirrors the IsaacLab-Arena
:mod:`keyboard` / :mod:`spacemouse` device modules.
"""

from __future__ import annotations

from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.gamepad import Se3GamepadCfg
from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


@register_device
class GamepadTeleopDevice(TeleopDeviceBase):
    """Teleop device backed by IsaacLab's Se(3) gamepad driver."""

    name = "gamepad"

    def __init__(
        self,
        sim_device: str | None = None,
        pos_sensitivity: float = 1.0,
        rot_sensitivity: float = 1.6,
        dead_zone: float = 0.01,
    ):
        super().__init__(sim_device=sim_device)
        self.pos_sensitivity = pos_sensitivity
        self.rot_sensitivity = rot_sensitivity
        self.dead_zone = dead_zone

    def get_teleop_device_cfg(self, embodiment: object | None = None) -> DevicesCfg:
        return DevicesCfg(
            devices={
                "gamepad": Se3GamepadCfg(
                    pos_sensitivity=self.pos_sensitivity,
                    rot_sensitivity=self.rot_sensitivity,
                    dead_zone=self.dead_zone,
                    sim_device=self.sim_device,
                ),
            }
        )

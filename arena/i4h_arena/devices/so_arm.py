# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM 101 teleoperation as Isaac Lab devices: the keyboard, and a leader arm on serial.

The scene declares both under ``teleop_devices``, so Isaac Lab's ``record_demos.py
--teleop_device keyboard`` or ``--teleop_device so101_leader`` drives the arm. Each
produces the six absolute joint targets the SO-ARM's joint-position actions take.
"""

from __future__ import annotations

from collections.abc import Callable

import numpy as np
import torch
from isaaclab.devices import Se3Keyboard, Se3KeyboardCfg
from isaaclab.devices.device_base import DeviceBase, DeviceCfg
from isaaclab.utils import configclass

from i4h_common.config import get_robot_config

_JAW_OPEN = 0.35
_JAW_CLOSED = -0.16


@configclass
class SoArmKeyboardCfg(DeviceCfg):
    """Keyboard joint control: the six Cartesian keys step the five arm joints, ``K`` toggles the jaw."""

    home_joint_pos_rad: tuple[float, ...] = ()
    """Joint targets the arm starts from and returns to on reset."""

    step_rad: float = 0.02
    """Joint step per keyboard tick [rad]."""

    sensitivity: float = 1.0
    class_type: type[DeviceBase] | str = "{DIR}.so_arm:SoArmKeyboard"


class SoArmKeyboard(DeviceBase):
    """Isaac Lab's SE(3) keyboard read as joint steps: its first five values move the five arm joints."""

    def __init__(self, cfg: SoArmKeyboardCfg, retargeters: list | None = None):
        super().__init__(retargeters)
        self.cfg = cfg
        self._keyboard = Se3Keyboard(
            Se3KeyboardCfg(pos_sensitivity=0.05 * cfg.sensitivity, rot_sensitivity=0.15 * cfg.sensitivity)
        )
        self._home = np.asarray(cfg.home_joint_pos_rad, dtype=np.float32)
        self._targets = self._home.copy()

    def __str__(self) -> str:
        return f"SO-ARM keyboard: {self._keyboard}"

    def reset(self) -> None:
        self._keyboard.reset()
        self._targets = self._home.copy()

    def add_callback(self, key: str, func: Callable) -> None:
        self._keyboard.add_callback(key, func)

    def advance(self) -> torch.Tensor:
        command = self._keyboard.advance().detach().cpu().numpy().reshape(-1)
        self._targets[:5] += command[:5] * self.cfg.step_rad
        self._targets[5] = _JAW_OPEN if command[6] > 0 else _JAW_CLOSED
        return torch.from_numpy(self._targets.copy()).to(self.cfg.sim_device)


@configclass
class SoArmLeaderCfg(DeviceCfg):
    """An SO-ARM 101 leader arm streaming comma-separated joint degrees over serial."""

    port: str = "/dev/ttyACM1"
    robot: str = "so101"
    """Robot descriptor holding the leader calibration (``arena/i4h_arena/embodiments/manifest``)."""

    class_type: type[DeviceBase] | str = "{DIR}.so_arm:SoArmLeader"


class SoArmLeader(DeviceBase):
    """The follower mirrors the leader: leader degrees mapped to sim radians by the robot's calibration.

    The calibration lives with the robot descriptor, not in this driver, because the
    dataset converter needs the same numbers.
    """

    def __init__(self, cfg: SoArmLeaderCfg, retargeters: list | None = None):
        super().__init__(retargeters)
        import serial  # noqa: PLC0415

        self.cfg = cfg
        self._robot = get_robot_config(cfg.robot)
        self._home = np.asarray(self._robot.home_joint_pos_rad, dtype=np.float32)
        self._targets = self._home.copy()
        try:
            self._serial = serial.Serial(cfg.port, baudrate=1_000_000, timeout=0.0)
        except Exception as exc:  # noqa: BLE001 - name the cable, whatever pyserial raised
            raise RuntimeError(f"cannot open the SO-ARM leader on {cfg.port}: {exc}. Check the cable.") from exc

    def __str__(self) -> str:
        return f"SO-ARM leader on {self.cfg.port}"

    def reset(self) -> None:
        pass

    def add_callback(self, key: str, func: Callable) -> None:
        pass

    def advance(self) -> torch.Tensor:
        # Non-blocking: no new line this step keeps the last targets.
        line = self._serial.readline()
        if line:
            try:
                degrees = np.asarray([float(value) for value in line.decode().strip().split(",")], dtype=np.float32)
            except (UnicodeDecodeError, ValueError):
                degrees = None
            if degrees is not None and degrees.size == self._home.size:
                self._targets = self.to_sim(degrees)
        return torch.from_numpy(self._targets.copy()).to(self.cfg.sim_device)

    def to_sim(self, leader_deg: np.ndarray) -> np.ndarray:
        """Leader degrees to sim radians, using the robot descriptor's calibration."""
        extra = self._robot.extra.get("so101_leader", {})
        signs = extra.get("sim_signs_by_joint", {})
        radians = np.deg2rad(leader_deg)
        for index, joint in enumerate(self._robot.joint_names):
            radians[index] *= float(signs.get(joint, 1.0))
        wrist_baseline = float(extra.get("wrist_flex_baseline_deg", 0.0))
        if wrist_baseline and "wrist_flex" in self._robot.joint_names:
            radians[self._robot.joint_index("wrist_flex")] -= np.deg2rad(wrist_baseline)
        return (radians + self._home).astype(np.float32)

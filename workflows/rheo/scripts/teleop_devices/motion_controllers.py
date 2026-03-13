# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from dataclasses import dataclass

import numpy as np
import torch
from isaaclab.devices.device_base import DevicesCfg
from isaaclab.devices.openxr import OpenXRDeviceCfg
from isaaclab.devices.openxr.retargeters import G1LowerBodyStandingMotionControllerRetargeterCfg
from isaaclab.devices.openxr.retargeters.humanoid.unitree.trihand.g1_upper_body_motion_ctrl_gripper import (
    G1TriHandUpperBodyMotionControllerGripperRetargeter,
    G1TriHandUpperBodyMotionControllerGripperRetargeterCfg,
)
from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab_arena.assets.register import register_device
from isaaclab_arena.teleop_devices.teleop_device_base import TeleopDeviceBase


class TrocarG1MotionControllerGripperRetargeter(G1TriHandUpperBodyMotionControllerGripperRetargeter):
    """Convert upstream absolute wrist poses into trocar robot-local targets."""

    _ROBOT_ORIGIN_W = np.array([-1.84919, 1.94, 0.81168], dtype=np.float32)

    def retarget(self, data: dict) -> torch.Tensor:
        action = super().retarget(data).clone()
        action[2:5] = action.new_tensor(action[2:5].detach().cpu().numpy() - self._ROBOT_ORIGIN_W)
        action[9:12] = action.new_tensor(action[9:12].detach().cpu().numpy() - self._ROBOT_ORIGIN_W)
        return action

    def reset(self):
        self._prev_left_state = 0.0
        self._prev_right_state = 0.0


@dataclass
class TrocarG1MotionControllerGripperRetargeterCfg(G1TriHandUpperBodyMotionControllerGripperRetargeterCfg):
    retargeter_type: type[G1TriHandUpperBodyMotionControllerGripperRetargeter] = (
        TrocarG1MotionControllerGripperRetargeter
    )


@register_device
class MotionControllersTeleopDevice(TeleopDeviceBase):
    """Teleop device for VR motion controllers (e.g., Quest controllers)."""

    name = "motion_controllers"

    def __init__(self, sim_device: str | None = None):
        super().__init__(sim_device=sim_device)

    def get_teleop_device_cfg(
        self,
        embodiment: object | None = None,
        xr_cfg: XrCfg | None = None,
        use_trocar_retargeter: bool = False,
    ) -> DevicesCfg:
        """Build the teleop device configuration (WBC: Gripper + LowerBodyStanding)."""

        if xr_cfg is None:
            if embodiment is not None and hasattr(embodiment, "get_xr_cfg"):
                xr_cfg = embodiment.get_xr_cfg()
            else:
                xr_cfg = XrCfg()

        xr_cfg.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        if use_trocar_retargeter:
            retargeters = [
                TrocarG1MotionControllerGripperRetargeterCfg(
                    sim_device=self.sim_device,
                ),
                G1LowerBodyStandingMotionControllerRetargeterCfg(
                    sim_device=self.sim_device,
                ),
            ]
        else:
            retargeters = [
                G1TriHandUpperBodyMotionControllerGripperRetargeterCfg(
                    sim_device=self.sim_device,
                ),
                G1LowerBodyStandingMotionControllerRetargeterCfg(
                    sim_device=self.sim_device,
                ),
            ]

        return DevicesCfg(
            devices={
                "motion_controllers": OpenXRDeviceCfg(
                    retargeters=retargeters,
                    sim_device=self.sim_device,
                    xr_cfg=xr_cfg,
                ),
            }
        )

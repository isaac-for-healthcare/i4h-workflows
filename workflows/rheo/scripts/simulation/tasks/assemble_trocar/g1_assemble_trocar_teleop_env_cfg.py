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


from isaaclab.devices.openxr.xr_cfg import XrAnchorRotationMode, XrCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import configclass
from isaaclab_arena_g1.g1_env.mdp import g1_events as g1_events_mdp
from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action import G1DecoupledWBCPinkAction
from simulation.tasks.assemble_trocar.g1_assemble_trocar_env_cfg import G1AssembleTrocarEnvCfg
from teleop_devices.motion_controllers import MotionControllersTeleopDevice

from isaaclab_arena_g1.g1_env.mdp.actions.g1_decoupled_wbc_pink_action_cfg import (  # isort: skip
    G1DecoupledWBCPinkActionCfg,
)


class G1AssembleTrocarFixedLegsWBCPinkAction(G1DecoupledWBCPinkAction):
    """Assemble trocar teleop action term with lower body hard-fixed to zero targets."""

    _FIXED_LEG_WAIST_JOINTS = (
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
    )
    _FIXED_BASE_HEIGHT_CMD = 0.75

    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        joint_names = self._asset.data.joint_names
        self._fixed_leg_waist_joint_ids = [joint_names.index(name) for name in self._FIXED_LEG_WAIST_JOINTS]

    def process_actions(self, actions):
        actions_fixed = actions.clone()

        # For assemble trocar we do not want the teleop stream to drive the lower body at all:
        # no navigation, nominal standing base height, and zero torso orientation command.
        nav_start = -self.navigate_cmd_dim - self.base_height_cmd_dim - self.torso_orientation_rpy_cmd_dim
        nav_end = -self.base_height_cmd_dim - self.torso_orientation_rpy_cmd_dim
        base_start = -self.base_height_cmd_dim - self.torso_orientation_rpy_cmd_dim
        base_end = -self.torso_orientation_rpy_cmd_dim
        torso_start = -self.torso_orientation_rpy_cmd_dim

        actions_fixed[:, nav_start:nav_end] = 0.0
        actions_fixed[:, base_start:base_end] = self._FIXED_BASE_HEIGHT_CMD
        actions_fixed[:, torso_start:] = 0.0

        super().process_actions(actions_fixed)
        self._processed_actions[:, self._fixed_leg_waist_joint_ids] = 0.0


@configclass
class G1AssembleTrocarFixedLegsWBCPinkActionCfg(G1DecoupledWBCPinkActionCfg):
    """Config for assemble trocar teleop action with fixed lower body."""

    class_type: type[ActionTerm] = G1AssembleTrocarFixedLegsWBCPinkAction


@configclass
class TeleopActionsCfg:
    """23-D WBC+PINK action: gripper(2) + wrist poses(14) + nav(3) + height(1) + torso(3)."""

    g1_action: ActionTermCfg = G1AssembleTrocarFixedLegsWBCPinkActionCfg(asset_name="robot", joint_names=[".*"])


@configclass
class G1AssembleTrocarTeleopEnvCfg(G1AssembleTrocarEnvCfg):
    """Meta Quest teleoperation variant of the assemble-trocar environment."""

    actions: TeleopActionsCfg = TeleopActionsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.70711, 0.0, 0.0, -0.70711),
    )

    def __post_init__(self):
        super().__post_init__()

        self.sim.render_interval = 2
        self.episode_length_s = 300.0

        self.events.reset_wbc_policy = EventTermCfg(
            func=g1_events_mdp.reset_decoupled_wbc_pink_policy,
            mode="reset",
        )

        # XR anchor follows the robot pelvis
        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        # Teleop devices in gripper mode, matching Arena's WBC+PINK pipeline.
        mc = MotionControllersTeleopDevice(sim_device=self.sim.device)
        self.teleop_devices = mc.get_teleop_device_cfg(xr_cfg=self.xr, use_trocar_retargeter=True)

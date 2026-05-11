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

"""H2 + Sharpa Wave teleop environment for the spread-tablecloth task.

Uses PinkIK for upper-body control (58D action: dual wrist SE3 + 44 Sharpa
hand joints) with OpenXR hand-tracking via IsaacTeleop.  The H2 lower body
is fixed.
"""

import os

import isaaclab.envs.mdp as base_mdp
from isaaclab.assets import ArticulationCfg
from isaaclab.controllers.pink_ik import FrameTaskCfg, NullSpacePostureTaskCfg, PinkIKControllerCfg
from isaaclab.envs.mdp.actions.pink_actions_cfg import PinkInverseKinematicsActionCfg
from isaaclab.managers import ActionTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.utils import configclass

from isaaclab_teleop.isaac_teleop_cfg import IsaacTeleopCfg
from isaaclab_teleop.xr_cfg import XrAnchorRotationMode, XrCfg

from .config import (
    CameraPresets,
    H2_DEFAULT_JOINT_POS,
    H2_SHARPA_CFG,
    H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER,
    H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    H2_SPREAD_TABLECLOTH_INIT_POS,
    H2_SPREAD_TABLECLOTH_INIT_ROT,
    _resolve_h2_urdf_path,
)
from .h2_spread_tablecloth_env_cfg import G1SpreadTableclothEnvCfg
from simulation.tasks.spread_tablecloth import mdp

# ---------------------------------------------------------------------------
# Sharpa Wave hand joint names (per-hand, in URDF order for DexPilot)
# ---------------------------------------------------------------------------
_LEFT_HAND_JOINT_NAMES = [
    "left_thumb_CMC_FE",
    "left_thumb_CMC_AA",
    "left_thumb_MCP_FE",
    "left_thumb_MCP_AA",
    "left_thumb_IP",
    "left_index_MCP_FE",
    "left_index_MCP_AA",
    "left_index_PIP",
    "left_index_DIP",
    "left_middle_MCP_FE",
    "left_middle_MCP_AA",
    "left_middle_PIP",
    "left_middle_DIP",
    "left_ring_MCP_FE",
    "left_ring_MCP_AA",
    "left_ring_PIP",
    "left_ring_DIP",
    "left_pinky_CMC",
    "left_pinky_MCP_FE",
    "left_pinky_MCP_AA",
    "left_pinky_PIP",
    "left_pinky_DIP",
]

_RIGHT_HAND_JOINT_NAMES = [
    "right_thumb_CMC_FE",
    "right_thumb_CMC_AA",
    "right_thumb_MCP_FE",
    "right_thumb_MCP_AA",
    "right_thumb_IP",
    "right_index_MCP_FE",
    "right_index_MCP_AA",
    "right_index_PIP",
    "right_index_DIP",
    "right_middle_MCP_FE",
    "right_middle_MCP_AA",
    "right_middle_PIP",
    "right_middle_DIP",
    "right_ring_MCP_FE",
    "right_ring_MCP_AA",
    "right_ring_PIP",
    "right_ring_DIP",
    "right_pinky_CMC",
    "right_pinky_MCP_FE",
    "right_pinky_MCP_AA",
    "right_pinky_PIP",
    "right_pinky_DIP",
]

# ---------------------------------------------------------------------------
# IsaacTeleop action-tensor layout (58D)
# [left_wrist_pos(3), left_wrist_quat(4),
#  right_wrist_pos(3), right_wrist_quat(4),
#  hand_joints(44) in PhysX BFS articulation order]
# ---------------------------------------------------------------------------
_LEFT_EE_ELEMENTS = [
    "l_pos_x", "l_pos_y", "l_pos_z", "l_quat_x", "l_quat_y", "l_quat_z", "l_quat_w",
]
_RIGHT_EE_ELEMENTS = [
    "r_pos_x", "r_pos_y", "r_pos_z", "r_quat_x", "r_quat_y", "r_quat_z", "r_quat_w",
]

_OUTPUT_ORDER = (
    _LEFT_EE_ELEMENTS
    + _RIGHT_EE_ELEMENTS
    + H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER
)


def _build_h2_sharpa_tablecloth_pipeline():
    """IsaacTeleop retargeting pipeline for H2 + Sharpa Wave (hand tracking from AVP).

    Se3AbsRetargeter (wrists) + DexHandRetargeter (Sharpa fingers) -> TensorReorderer -> 58D action.
    """
    from isaacteleop.retargeters import (
        DexHandRetargeter,
        DexHandRetargeterConfig,
        Se3AbsRetargeter,
        Se3RetargeterConfig,
        TensorReorderer,
    )
    from isaacteleop.retargeting_engine.deviceio_source_nodes import HandsSource
    from isaacteleop.retargeting_engine.interface import OutputCombiner, ValueInput
    from isaacteleop.retargeting_engine.tensor_types import TransformMatrix

    hands = HandsSource(name="hands")
    transform_input = ValueInput("world_T_anchor", TransformMatrix())
    transformed_hands = hands.transformed(transform_input.output(ValueInput.VALUE))

    # Wrist SE3 retargeters with H2-specific offsets (from isaaclab_arena_h2 h2_pink_pipeline)
    left_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.LEFT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=45.0,
        target_offset_pitch=180.0,
        target_offset_yaw=-90.0,
    )
    left_se3 = Se3AbsRetargeter(left_se3_cfg, name="left_ee_pose")
    connected_left_se3 = left_se3.connect(
        {HandsSource.LEFT: transformed_hands.output(HandsSource.LEFT)}
    )

    right_se3_cfg = Se3RetargeterConfig(
        input_device=HandsSource.RIGHT,
        zero_out_xy_rotation=False,
        use_wrist_rotation=True,
        use_wrist_position=True,
        target_offset_roll=-135.0,
        target_offset_pitch=0.0,
        target_offset_yaw=90.0,
    )
    right_se3 = Se3AbsRetargeter(right_se3_cfg, name="right_ee_pose")
    connected_right_se3 = right_se3.connect(
        {HandsSource.RIGHT: transformed_hands.output(HandsSource.RIGHT)}
    )

    # DexPilot retargeters for Sharpa Wave hands
    _h2_pkg = "/home/mxgu/Workspace/Omniverse/gmx/IsaacLab-Arena/isaaclab_arena_h2"

    _config_dir = os.path.join(_h2_pkg, "teleop", "data", "configs")
    left_yaml_path = os.path.join(_config_dir, "sharpa_wave_left_dexpilot.yml")
    right_yaml_path = os.path.join(_config_dir, "sharpa_wave_right_dexpilot.yml")

    _assets_dir = os.path.join(_h2_pkg, "assets", "urdf", "sharpa_standalone")
    left_hand_urdf = os.path.join(_assets_dir, "left_sharpa_wave.urdf")
    right_hand_urdf = os.path.join(_assets_dir, "right_sharpa_wave.urdf")

    operator2mano = (0, -1, 0, -1, 0, 0, 0, 0, -1)

    left_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=left_yaml_path,
            hand_urdf=left_hand_urdf,
            hand_joint_names=_LEFT_HAND_JOINT_NAMES,
            hand_side="left",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="left_hand",
    )
    connected_left_dex = left_dex.connect(
        {HandsSource.LEFT: hands.output(HandsSource.LEFT)}
    )

    right_dex = DexHandRetargeter(
        DexHandRetargeterConfig(
            hand_retargeting_config=right_yaml_path,
            hand_urdf=right_hand_urdf,
            hand_joint_names=_RIGHT_HAND_JOINT_NAMES,
            hand_side="right",
            handtracking_to_baselink_frame_transform=operator2mano,
        ),
        name="right_hand",
    )
    connected_right_dex = right_dex.connect(
        {HandsSource.RIGHT: hands.output(HandsSource.RIGHT)}
    )

    reorderer = TensorReorderer(
        input_config={
            "left_ee_pose": _LEFT_EE_ELEMENTS,
            "right_ee_pose": _RIGHT_EE_ELEMENTS,
            "left_hand_joints": _LEFT_HAND_JOINT_NAMES,
            "right_hand_joints": _RIGHT_HAND_JOINT_NAMES,
        },
        output_order=_OUTPUT_ORDER,
        name="action_reorderer",
        input_types={
            "left_ee_pose": "array",
            "right_ee_pose": "array",
            "left_hand_joints": "scalar",
            "right_hand_joints": "scalar",
        },
    )
    connected_reorderer = reorderer.connect({
        "left_ee_pose": connected_left_se3.output("ee_pose"),
        "right_ee_pose": connected_right_se3.output("ee_pose"),
        "left_hand_joints": connected_left_dex.output("hand_joints"),
        "right_hand_joints": connected_right_dex.output("hand_joints"),
    })

    return OutputCombiner({"action": connected_reorderer.output("output")})


# ---------------------------------------------------------------------------
# H2 observation config (no wrist cameras — H2 only has a head camera)
# ---------------------------------------------------------------------------
@configclass
class H2ObservationsCfg:
    """Observation configuration for H2 spread-tablecloth teleop."""

    @configclass
    class PolicyCfg(ObsGroup):
        robot_joint_state = ObsTerm(func=mdp.get_robot_joint_states)

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    @configclass
    class CameraImagesCfg(ObsGroup):
        front_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


# ---------------------------------------------------------------------------
# PinkIK action config for H2 (58D)
# ---------------------------------------------------------------------------
@configclass
class H2TeleopActionsCfg:
    """58-D PINK IK action for OpenXR teleop with Sharpa Wave hands."""

    pink_ik_cfg: ActionTermCfg = PinkInverseKinematicsActionCfg(
        pink_controlled_joint_names=[
            "left_shoulder_pitch_joint",
            "left_shoulder_roll_joint",
            "left_shoulder_yaw_joint",
            "left_elbow_joint",
            "left_wrist_roll_joint",
            "left_wrist_pitch_joint",
            "left_wrist_yaw_joint",
            "right_shoulder_pitch_joint",
            "right_shoulder_roll_joint",
            "right_shoulder_yaw_joint",
            "right_elbow_joint",
            "right_wrist_roll_joint",
            "right_wrist_pitch_joint",
            "right_wrist_yaw_joint",
        ],
        hand_joint_names=H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER,
        target_eef_link_names={
            "left_wrist": "left_wrist_yaw_link",
            "right_wrist": "right_wrist_yaw_link",
        },
        asset_name="robot",
        controller=PinkIKControllerCfg(
            articulation_name="robot",
            base_link_name="pelvis",
            num_hand_joints=44,
            show_ik_warnings=True,
            fail_on_joint_limit_violation=False,
            variable_input_tasks=[
                FrameTaskCfg(
                    frame="left_wrist_yaw_link",
                    position_cost=1.0,
                    orientation_cost=0.5,
                ),
                FrameTaskCfg(
                    frame="right_wrist_yaw_link",
                    position_cost=1.0,
                    orientation_cost=0.5,
                ),
                NullSpacePostureTaskCfg(
                    cost=0.01,
                    lm_damping=1.0,
                    controlled_frames=[
                        "left_wrist_yaw_link",
                        "right_wrist_yaw_link",
                    ],
                    controlled_joints=[
                        "left_shoulder_pitch_joint",
                        "left_shoulder_roll_joint",
                        "left_shoulder_yaw_joint",
                        "left_elbow_joint",
                        "right_shoulder_pitch_joint",
                        "right_shoulder_roll_joint",
                        "right_shoulder_yaw_joint",
                        "right_elbow_joint",
                    ],
                ),
            ],
            fixed_input_tasks=[],
        ),
        enable_gravity_compensation=False,
    )


# ---------------------------------------------------------------------------
# Env config
# ---------------------------------------------------------------------------
@configclass
class H2SpreadTableclothTeleopEnvCfg(G1SpreadTableclothEnvCfg):
    """OpenXR teleop variant with H2 + Sharpa Wave dexterous hands.

    Inherits scene (table, cloth, lights) and MDP (observations, terminations,
    events) from the base G1 spread-tablecloth config.  Replaces the robot
    articulation with H2 and the action config with 58D PinkIK.
    """

    actions: H2TeleopActionsCfg = H2TeleopActionsCfg()
    observations: H2ObservationsCfg = H2ObservationsCfg()

    xr: XrCfg = XrCfg(
        anchor_pos=(0.0, 0.0, -1.0),
        anchor_rot=(0.0, 0.0, -0.70711, 0.70711),
    )

    def __post_init__(self):
        super().__post_init__()

        _joint_pos = H2_DEFAULT_JOINT_POS.copy()
        _joint_pos.update(H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS)

        self.scene.robot = H2_SHARPA_CFG.replace(
            prim_path="/World/envs/env_.*/Robot",
            init_state=ArticulationCfg.InitialStateCfg(
                pos=H2_SPREAD_TABLECLOTH_INIT_POS,
                rot=H2_SPREAD_TABLECLOTH_INIT_ROT,
                joint_pos=_joint_pos,
                joint_vel={".*": 0.0},
            ),
        )

        self.scene.front_camera = CameraPresets.h2_front_camera(focal_length=10.5)
        self.scene.left_wrist_camera = None
        self.scene.right_wrist_camera = None

        self.episode_length_s = 300.0

        h2_urdf_path = _resolve_h2_urdf_path()
        self.actions.pink_ik_cfg.controller.urdf_path = h2_urdf_path
        self.actions.pink_ik_cfg.controller.mesh_path = os.path.dirname(h2_urdf_path)

        self.xr.anchor_prim_path = "/World/envs/env_0/Robot/pelvis"
        self.xr.fixed_anchor_height = True
        self.xr.anchor_rotation_mode = XrAnchorRotationMode.FOLLOW_PRIM_SMOOTHED

        self.isaac_teleop = IsaacTeleopCfg(
            pipeline_builder=_build_h2_sharpa_tablecloth_pipeline,
            sim_device=self.sim.device,
            xr_cfg=self.xr,
        )

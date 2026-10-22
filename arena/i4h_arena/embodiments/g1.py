# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 Arena embodiments.

Importing this module registers:

* ``g1_locomanip_wbc_joint`` / ``g1_locomanip_wbc_pink``: IsaacLab-Arena's
  ``g1_wbc_joint`` / ``g1_wbc_pink`` on Rheo's USD, with an RGB head-camera
  policy observation, action/state recorders and pelvis-frame Mimic poses.
* :class:`G1AssembleTrocarJointEmbodiment` as ``g1_assemble_trocar_joint`` for
  the Assemble Trocar task.

Loaded as a side-effect import from the humanoid envs' ``register_assets``.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from typing import Any

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
import isaaclab.utils.math as PoseUtils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers.action_manager import ActionTermCfg
from isaaclab.sensors import CameraCfg
from isaaclab.utils import configclass
from isaaclab_arena.assets.register import register_asset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.embodiments.g1.g1 import *  # noqa: F401,F403
from isaaclab_arena.embodiments.g1.g1 import G1MimicEnv, G1SceneCfg, G1WBCJointEmbodiment, G1WBCPinkEmbodiment
from isaaclab_arena.utils.cameras import ArenaCameraCfg
from isaaclab_arena.utils.pose import Pose

from i4h_arena.assets.constants import UNITREE_G1_29DOF_BASE_FIX_USD, UNITREE_G1_29DOF_USD
from i4h_arena.tensor_utils import to_torch

# NOTE: keep this import AFTER `from isaaclab_arena.embodiments.g1.g1 import *`
# above. That star import re-exports a module-level ``mdp``
# (``isaaclab_tasks…pick_place.mdp``, which has no ``__all__``); importing
# ``g1_trocar as mdp`` earlier lets the star import clobber this binding,
# breaking ``mdp.ASSEMBLE_TROCAR_JOINT_NAMES`` below. Must stay last.
from i4h_arena.envcfg import g1_trocar as mdp  # isort: skip

# ---------- Locomanip path: Arena's stock WBC embodiments on Rheo's G1. ---------


class G1LocomanipMimicEnv(G1MimicEnv):
    """Arena's G1 Mimic env with object poses expressed in the pelvis frame."""

    def get_object_poses(self, env_ids: Sequence[int] | None = None):
        if env_ids is None:
            env_ids = slice(None)
        pelvis_pose_w = to_torch(self.scene["robot"].data.body_link_state_w)[
            :, self.scene["robot"].data.body_names.index("pelvis"), :
        ]
        pelvis_position_w = pelvis_pose_w[:, :3] - self.scene.env_origins
        pelvis_rot_mat_w = PoseUtils.matrix_from_quat(pelvis_pose_w[:, 3:7])
        pelvis_pose_mat_w = PoseUtils.make_pose(pelvis_position_w, pelvis_rot_mat_w)
        pelvis_pose_inv = PoseUtils.pose_inv(pelvis_pose_mat_w)
        state = self.scene.get_state(is_relative=True)
        object_pose_matrix: dict[str, Any] = {}
        for group_name in ("rigid_object", "articulation"):
            if group_name not in state:
                continue
            for obj_name, obj_state in state[group_name].items():
                object_pose_mat_w = PoseUtils.make_pose(
                    obj_state["root_pose"][env_ids, :3],
                    PoseUtils.matrix_from_quat(obj_state["root_pose"][env_ids, 3:7]),
                )
                object_pose_matrix[obj_name] = torch.matmul(pelvis_pose_inv, object_pose_mat_w)
        return object_pose_matrix


class _G1Locomanip:
    """Rheo's G1 USD, an RGB head-camera policy observation and action/state recorders.

    Mixed in ahead of an Arena G1 WBC embodiment.
    """

    def __init__(self, *args: Any, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        # Isaac Lab 3's Camera already renders tiled; TiledCameraCfg is deprecated.
        self.camera_config.set_use_tiled_camera(False)
        self.mimic_env = G1LocomanipMimicEnv

    def get_scene_cfg(self):
        scene_cfg = super().get_scene_cfg()
        if getattr(scene_cfg, "robot", None) is not None:
            scene_cfg.robot = scene_cfg.robot.copy()
            scene_cfg.robot.spawn.usd_path = UNITREE_G1_29DOF_USD
            scene_cfg.robot.spawn.semantic_tags = [("class", "robot")]
        return scene_cfg

    def get_observation_cfg(self):
        obs_cfg = super().get_observation_cfg()
        if self.enable_cameras and hasattr(obs_cfg, "policy") and not hasattr(obs_cfg.policy, "robot_head_cam"):
            obs_cfg.policy.robot_head_cam = ObsTerm(
                func=base_mdp.image_rgb,
                params={
                    "sensor_cfg": SceneEntityCfg("robot_head_cam"),
                    "data_type": "rgb",
                    "normalize": False,
                },
            )
        return obs_cfg

    def get_recorder_term_cfg(self, record_trajectories: bool = False):
        # Arena's G1 records nothing per step, which leaves IsaacLab's demo groups
        # without actions, observations or states. Its observations name the joints
        # robot_joint_pos, so the converter's obs/joint_pos is recorded on its own.
        from i4h_arena.embodiments.recorders import action_state_recorders, with_joint_positions

        return with_joint_positions(action_state_recorders(super().get_recorder_term_cfg(record_trajectories)))


@register_asset
class G1LocomanipWBCJointEmbodiment(_G1Locomanip, G1WBCJointEmbodiment):
    """Arena's ``g1_wbc_joint`` on Rheo's G1, for the locomanipulation policies."""

    name = "g1_locomanip_wbc_joint"
    tags = ["embodiment", "g1"]


@register_asset
class G1LocomanipWBCPinkEmbodiment(_G1Locomanip, G1WBCPinkEmbodiment):
    """Arena's ``g1_wbc_pink`` on Rheo's G1, for locomanipulation teleoperation."""

    name = "g1_locomanip_wbc_pink"
    tags = ["embodiment", "g1"]


# ---------- Assemble Trocar path: a custom G1-only direct-joint embodiment. ----


def _camera_cfg(
    *,
    prim_path: str,
    focal_length: float,
    pos_offset: tuple[float, float, float],
    rot_offset: tuple[float, float, float, float],
) -> CameraCfg:
    return CameraCfg(
        prim_path=prim_path,
        update_period=0.02,
        height=480,
        width=640,
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=focal_length,
            focus_distance=400.0,
            horizontal_aperture=20.0,
            clipping_range=(0.1, 1.0e5),
        ),
        offset=CameraCfg.OffsetCfg(pos=pos_offset, rot=rot_offset, convention="ros"),
    )


@configclass
class G1AssembleTrocarActionsCfg:
    """43-D direct joint position action used by the original Assemble Trocar model."""

    joint_pos: ActionTermCfg = base_mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=mdp.ASSEMBLE_TROCAR_JOINT_NAMES,
        scale=1.0,
        use_default_offset=False,
        offset=mdp.ASSEMBLE_TROCAR_ACTION_OFFSETS,
        preserve_order=True,
    )


@configclass
class G1AssembleTrocarObservationsCfg:
    @configclass
    class PolicyCfg(ObsGroup):
        assemble_trocar_joint_pos = ObsTerm(func=mdp.get_assemble_trocar_joint_positions)
        robot_joint_state = ObsTerm(func=mdp.get_robot_body_joint_states)
        robot_dex3_joint_state = ObsTerm(func=mdp.get_robot_dex3_joint_states)
        front_camera_rgb = ObsTerm(
            func=base_mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("front_camera"), "data_type": "rgb", "normalize": False},
        )
        left_wrist_camera_rgb = ObsTerm(
            func=base_mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera_rgb = ObsTerm(
            func=base_mdp.image_rgb,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.enable_corruption = False
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()


@configclass
class G1AssembleTrocarCameraCfg(ArenaCameraCfg):
    front_camera = _camera_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/d435_link/front_cam",
        focal_length=10.5,
        pos_offset=(0.0, 0.0, 0.0),
        rot_offset=(-0.5, 0.5, -0.5, 0.5),
    )
    left_wrist_camera = _camera_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/left_hand_camera_base_link/left_wrist_camera",
        focal_length=12.0,
        pos_offset=(-0.04012, -0.07441, 0.15711),
        rot_offset=(0.86024, 0.0424, 0.50809, 0.00539),
    )
    right_wrist_camera = _camera_cfg(
        prim_path="{ENV_REGEX_NS}/Robot/right_hand_camera_base_link/right_wrist_camera",
        focal_length=12.0,
        pos_offset=(-0.04012, 0.07441, 0.15711),
        rot_offset=(0.86024, 0.0424, 0.50809, 0.00539),
    )


def _assemble_trocar_scene_cfg():
    scene_cfg = copy.deepcopy(G1SceneCfg())
    robot = scene_cfg.robot.copy()
    robot.spawn.usd_path = UNITREE_G1_29DOF_BASE_FIX_USD
    robot.init_state = ArticulationCfg.InitialStateCfg(
        pos=(-1.84919, 1.94, 0.81168),
        rot=(0.0, 0.0, 0.0, 1.0),
        joint_pos={
            "left_hip_pitch_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_yaw_joint": 0.0,
            "left_knee_joint": 0.0,
            "left_ankle_pitch_joint": 0.0,
            "left_ankle_roll_joint": 0.0,
            "right_hip_pitch_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_knee_joint": 0.0,
            "right_ankle_pitch_joint": 0.0,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": -0.754599,
            "left_shoulder_roll_joint": 0.550010,
            "left_shoulder_yaw_joint": -0.399298,
            "left_elbow_joint": 0.278886,
            "left_wrist_roll_joint": 0.320559,
            "left_wrist_pitch_joint": -0.203525,
            "left_wrist_yaw_joint": -0.387435,
            "right_shoulder_pitch_joint": -0.340858,
            "right_shoulder_roll_joint": -0.186152,
            "right_shoulder_yaw_joint": 0.015023,
            "right_elbow_joint": -0.777159,
            "right_wrist_roll_joint": 0.019805,
            "right_wrist_pitch_joint": 1.182285,
            "right_wrist_yaw_joint": -0.022848,
            "left_hand_index_0_joint": -1.0471975512,
            "left_hand_middle_0_joint": -1.0471975512,
            "left_hand_thumb_0_joint": 0.0,
            "left_hand_index_1_joint": -0.6981317008,
            "left_hand_middle_1_joint": -0.6981317008,
            "left_hand_thumb_1_joint": 0.0,
            "left_hand_thumb_2_joint": 0.0,
            "right_hand_index_0_joint": 1.0471975512,
            "right_hand_middle_0_joint": 1.0471975512,
            "right_hand_thumb_0_joint": 0.0,
            "right_hand_index_1_joint": 0.6981317008,
            "right_hand_middle_1_joint": 0.6981317008,
            "right_hand_thumb_1_joint": 0.0,
            "right_hand_thumb_2_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    )
    robot.soft_joint_pos_limit_factor = 0.9
    robot.actuators["waist"] = ImplicitActuatorCfg(
        joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
        effort_limit_sim=1000.0,
        stiffness={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
        damping={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
        armature=None,
    )
    robot.actuators["arms"].damping = {
        ".*_shoulder_pitch_joint": 15.0,
        ".*_shoulder_roll_joint": 15.0,
        ".*_shoulder_yaw_joint": 8.0,
        ".*_elbow_joint": 8.0,
        ".*_wrist_.*_joint": 4.0,
    }
    robot.actuators["hands"].stiffness = 8.0
    robot.actuators["hands"].damping = 1.5
    robot.actuators["hands"].friction = 0.5
    scene_cfg.robot = robot
    return scene_cfg


@register_asset
class G1AssembleTrocarJointEmbodiment(EmbodimentBase):
    """Arena embodiment for the base-fixed G1 Dex3 direct-joint Assemble Trocar policy."""

    name = "g1_assemble_trocar_joint"
    tags = ["embodiment", "g1"]

    def __init__(self, enable_cameras: bool = False, initial_pose: Pose | None = None):
        super().__init__(enable_cameras=enable_cameras, initial_pose=initial_pose)
        self.scene_config = _assemble_trocar_scene_cfg()
        self.camera_config = G1AssembleTrocarCameraCfg()
        # Isaac Lab 3's Camera already renders tiled; TiledCameraCfg is deprecated.
        self.camera_config.set_use_tiled_camera(False)
        self.action_config = G1AssembleTrocarActionsCfg()
        self.observation_config = G1AssembleTrocarObservationsCfg()
        if not enable_cameras:
            # Without camera sensors (Isaac Lab's replay_demos, joint-only runs) observe joints only.
            for term in ("front_camera_rgb", "left_wrist_camera_rgb", "right_wrist_camera_rgb"):
                setattr(self.observation_config.policy, term, None)

    def get_recorder_term_cfg(self, record_trajectories: bool = False):
        from i4h_arena.embodiments.recorders import action_state_recorders, with_joint_positions
        from i4h_arena.envcfg.g1_trocar import ASSEMBLE_TROCAR_JOINT_NAMES

        # In the checkpoint's 29-body + 14-Dex3 order, not the USD's topology order.
        return with_joint_positions(
            action_state_recorders(super().get_recorder_term_cfg(record_trajectories)),
            joint_names=ASSEMBLE_TROCAR_JOINT_NAMES,
        )

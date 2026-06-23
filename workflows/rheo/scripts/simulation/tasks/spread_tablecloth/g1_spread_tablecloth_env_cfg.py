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

"""G1 29DOF + Inspire env for the spread-tablecloth task."""

import os

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from simulation.tasks.spread_tablecloth import mdp
from simulation.tasks.spread_tablecloth.cloth_physics import make_newton_physics

# Local composition wrappers that split the bundled tablecloth USD into a
# cloth-only deformable and a standalone inner rigid body (Newton migration).
# The original omniverse:// asset is referenced unchanged by these wrappers.
_ASSET_DIR = os.path.join(os.path.dirname(__file__), "assets")
CLOTH_ONLY_USD = os.path.join(_ASSET_DIR, "cloth_only.usda")
CLOTH_INNER_USD = os.path.join(_ASSET_DIR, "cloth_inner.usda")

# Table asset (local, from surgery-room-dev-internal). We replace the original
# scene04.usd bundle with a plain ground plane + this table: scene04.usd carries a
# broken ``Cloth_fold007`` payload (missing on the server) which Newton's strict
# composition check (``parse_usd`` -> ``_raise_on_stage_errors``) rejects, whereas
# PhysX/Kit only warned. All other object positions are kept unchanged.
# FIXME: switch to the public/released asset URL once available.
TABLE_USD = "/home/shlabws1/workspace/surgery-room-dev-internal/assets/Assets/Assets/Table256/Table256.usd"
# Table256 is 1.0 x 0.6 x 0.77 m with its origin at the bbox center (tabletop
# slab +0.385 m above origin). Default scale: tabletop top is at z = 0.385 + 0.385
# = 0.77 m.
from simulation.tasks.spread_tablecloth.config import (
    SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    SPREAD_TABLECLOTH_INIT_POS,
    SPREAD_TABLECLOTH_INIT_ROT,
    CameraPresets,
    G1RobotPresets,
)

# G1 29 DOF body + Inspire hand joints.
joint_names = [
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
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
    "L_index_proximal_joint",
    "L_index_intermediate_joint",
    "L_middle_proximal_joint",
    "L_middle_intermediate_joint",
    "L_pinky_proximal_joint",
    "L_pinky_intermediate_joint",
    "L_ring_proximal_joint",
    "L_ring_intermediate_joint",
    "L_thumb_proximal_yaw_joint",
    "L_thumb_proximal_pitch_joint",
    "L_thumb_intermediate_joint",
    "L_thumb_distal_joint",
    "R_index_proximal_joint",
    "R_index_intermediate_joint",
    "R_middle_proximal_joint",
    "R_middle_intermediate_joint",
    "R_pinky_proximal_joint",
    "R_pinky_intermediate_joint",
    "R_ring_proximal_joint",
    "R_ring_intermediate_joint",
    "R_thumb_proximal_yaw_joint",
    "R_thumb_proximal_pitch_joint",
    "R_thumb_intermediate_joint",
    "R_thumb_distal_joint",
]


@configclass
class SpreadTableclothSceneCfg(InteractiveSceneCfg):
    """G1 robot + front/wrist cameras + scene + cloth + light."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_inspire_base_fix(
        init_pos=SPREAD_TABLECLOTH_INIT_POS,
        init_rot=SPREAD_TABLECLOTH_INIT_ROT,
        custom_joint_pos=SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    )

    front_camera = CameraPresets.g1_front_camera(focal_length=10.5)
    left_wrist_camera = CameraPresets.left_inspire_wrist_camera()
    right_wrist_camera = CameraPresets.right_inspire_wrist_camera()

    # Ground plane the robot stands on. World origin z=0 is at floor level
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # Table the cloth is spread on. 
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.43, 0.0, 0.385),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tablecloth",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.77),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            # Cloth-only composition wrapper. See cloth_only.usda.
            usd_path=CLOTH_ONLY_USD,
        ),
    )

    # Inner rigid body that the cloth wraps around.
    cloth_inner = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ClothInner",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CLOTH_INNER_USD,
            # Kinematic: cloth_inner spawns already penetrating the cloth/table
            # (the cloth's rest geometry is authored draped over it). As a free
            # dynamic body the Newton MJWarp solver blows up on that initial
            # penetration -> first-step NaN that propagates through the coupled
            # MJWarp+VBD solver. Kinematic keeps it as a fixed form the cloth
            # wraps around, without being physically solved.
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.83),
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1000.0,
        ),
    )


@configclass
class ActionsCfg:
    """Direct joint angle control."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Joint state + front/left/right wrist cameras."""

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
        left_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("left_wrist_camera"), "data_type": "rgb", "normalize": False},
        )
        right_wrist_camera = ObsTerm(
            func=base_mdp.image,
            params={"sensor_cfg": SceneEntityCfg("right_wrist_camera"), "data_type": "rgb", "normalize": False},
        )

        def __post_init__(self):
            self.concatenate_terms = False

    policy: PolicyCfg = PolicyCfg()
    camera_images: CameraImagesCfg = CameraImagesCfg()


@configclass
class TerminationsCfg:
    """Time-out only."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Reset scene + explicitly reset the standalone Cloth_In002 rigid body back to
    its (cloth-wrapped) initial pose each episode (otherwise it drifts/falls away)."""

    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")

    # ``cloth_asset_name`` points at the standalone ``cloth_inner`` asset (not the
    # deformable ``cloth``): reset_cloth_inner reads that asset's prim_path
    # (/World/envs/env_*/ClothInner) and globs ``<prim>/Cloth_In002/Cloth_In002``.
    # reset_cloth_inner = EventTermCfg(
    #     func=mdp.reset_cloth_inner,
    #     mode="reset",
    #     params={
    #         "cloth_asset_name": "cloth_inner",
    #         "inner_rel_path": "Cloth_In002/Cloth_In002",
    #     },
    # )


@configclass
class G1SpreadTableclothEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree G1 spread-tablecloth env."""

    scene: SpreadTableclothSceneCfg = SpreadTableclothSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=False,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(1.0, 0.8, 1.5),
        lookat=(-0.6, 0.5, 0.70),
        cam_prim_path="/OmniverseKit_Persp",
    )

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    terminations: TerminationsCfg = TerminationsCfg()
    events: EventCfg = EventCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2
        # Newton coupled MJWarp (robot) + VBD (cloth) backend.
        self.sim.physics = make_newton_physics()

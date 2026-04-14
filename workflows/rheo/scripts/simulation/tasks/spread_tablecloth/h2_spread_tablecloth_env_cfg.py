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

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import ArticulationCfg, AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_physx.physics import PhysxCfg
from simulation.tasks.spread_tablecloth import mdp
from isaaclab_physx.assets import DeformableObject, DeformableObjectCfg
from isaaclab_physx.sim import DeformableBodyPropertiesCfg, SurfaceDeformableBodyMaterialCfg

from simulation.tasks.spread_tablecloth.config import G1RobotPresets, CameraPresets  # isort: skip


TABLE_USD_PATH = (
    "/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/Assets/Table256/Table256.usd"
)
TABLECLOTH_USD_PATH = (
    "/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/Assets/Cloth/Cloth_fold02/Cloth_fold02.usd"
)
SCENE_USD_PATH = (
    "/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/scene05.usd"
)
TABLE_POS = (-0.50, 0.0, 0.385*0.9)
TABLE_ROT = (0.0, 0.0, 0.7071, 0.7071)
TABLE_SCALE = (0.6, 0.6, 0.9)
TABLE_TOP_SIZE = (1.2, 0.6, 0.04)
TABLE_TOP_POS = (-0.50, 0.0, 0.78*0.9)

# G1 29 DOF body + Dex3 hands.
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
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

@configclass
class SpreadTableclothSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the spread_tablecloth task (G1 robot + table + lights)."""

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(-0.95, 0.0, 0.80), init_rot=(0.0, 0.0, 0.0, 1.0)
    )

    front_camera = CameraPresets.g1_front_camera(focal_length=10.5)

    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCENE_USD_PATH,
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.9, -2.5, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )


    # ground = AssetBaseCfg(
    #     prim_path="/World/GroundPlane",
    #     spawn=sim_utils.GroundPlaneCfg(),
    # )

    # table = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Table",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_POS, rot=TABLE_ROT),
    #     spawn=sim_utils.UsdFileCfg(
    #         usd_path=TABLE_USD_PATH,
    #         scale=TABLE_SCALE,
    #         # The table USD ships with convex mesh colliders that are not
    #         # compatible with GPU cloth contact, so keep it visual-only.
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=False),
    #     ),
    # )
    # table_top_collider = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/TableTopCollider",
    #     init_state=AssetBaseCfg.InitialStateCfg(pos=TABLE_TOP_POS, rot=TABLE_ROT),
    #     spawn=sim_utils.CuboidCfg(
    #         size=TABLE_TOP_SIZE,
    #         collision_props=sim_utils.CollisionPropertiesCfg(collision_enabled=True),
    #         visual_material=sim_utils.PreviewSurfaceCfg(
    #             diffuse_color=(0.55, 0.55, 0.55),
    #         ),
    #     ),
    # )


    tablecloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tablecloth",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.65, 0.0, 0.81),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLECLOTH_USD_PATH,
            # deformable_props=DeformableBodyPropertiesCfg(disable_gravity=False),
            physics_material=SurfaceDeformableBodyMaterialCfg(
                density=100.0,
                youngs_modulus=5e5,
                poissons_ratio=0.1,
                surface_stretch_stiffness=1.0,
                surface_shear_stiffness=5000,
                surface_bend_stiffness=1,
            ),
        ),
    )

    # tablecloth = AssetBaseCfg(
    #     prim_path="{ENV_REGEX_NS}/Tablecloth",
    #     init_state=AssetBaseCfg.InitialStateCfg(
    #         pos=(-0.65, 0.0, 2.20),
    #         rot=(0.0, 0.0, 1.0, 0.0),
    #     ),
    #     spawn=sim_utils.UsdFileCfg(usd_path=TABLECLOTH_USD_PATH),
    # )

    light = AssetBaseCfg(
        prim_path="/World/light",
        spawn=sim_utils.DomeLightCfg(
            color=(0.75, 0.75, 0.75),
            intensity=1000.0,
        ),
    )


##
# MDP settings
##
@configclass
class ActionsCfg:
    """Direct joint angle control for G1 (29 DOF + Dex3 hands)."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=joint_names,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class ObservationsCfg:
    """Observation configuration for the spread_tablecloth task."""

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


@configclass
class TerminationsCfg:
    """Termination conditions — only time-out for now."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class EventCfg:
    """Event configuration for scene reset."""

    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class G1SpreadTableclothEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree G1 robot spread tablecloth environment configuration.

    Inherits from ManagerBasedRLEnvCfg, defines all configuration parameters
    for the spread_tablecloth environment.
    """

    scene: SpreadTableclothSceneCfg = SpreadTableclothSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=False,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.8, 3.0),
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
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
            gpu_max_deformable_surface_contacts=2**23,
        )


# Backward-compatible alias for any old imports.
H2SpreadTableclothEnvCfg = G1SpreadTableclothEnvCfg

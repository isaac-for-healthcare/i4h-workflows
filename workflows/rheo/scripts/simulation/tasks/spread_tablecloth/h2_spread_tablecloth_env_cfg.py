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
    "/workspaces/surgery-room-dev-internal/assets/Assets/Assets/Table256/Table256.usd"
)
TABLECLOTH_USD_PATH = (
    "/workspaces/surgery-room-dev-internal/assets/Assets/Assets/Cloth/Cloth_fold/Cloth_Out_Test.usd"
)

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

    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot: ArticulationCfg = G1RobotPresets.g1_29dof_dex3_base_fix(
        init_pos=(-0.85, 0.0, 0.80), init_rot=(0.0, 0.0, 0.0, 1.0)
    )

    front_camera = CameraPresets.g1_front_camera(focal_length=10.5)

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.40, 0.0, 0.385), rot=(0.0, 0.0, 0.7071, 0.7071)),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD_PATH,
            scale=(1.0, 1.0, 1.1),
        ),
    )

    tablecloth = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tablecloth",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.45, 0.0, 0.85),
            rot=(0.0, 0.0, 1.0, 0.0),
        ),
        spawn=sim_utils.UsdFileCfg(
                usd_path=TABLECLOTH_USD_PATH,
                # deformable_props=DeformableBodyPropertiesCfg(disable_gravity=False),
                physics_material=SurfaceDeformableBodyMaterialCfg(
                    density=100.0,
                    youngs_modulus=1e4,
                    poissons_ratio=0.4,
                    surface_stretch_stiffness=1.0,
                    surface_shear_stiffness=1.0,
                    surface_bend_stiffness=0.01,
                ),
        ),
    )

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
        self.sim.dt = 1 / 200
        self.sim.render_interval = self.decimation
        self.sim.physics = PhysxCfg(bounce_threshold_velocity=0.01)


# Backward-compatible alias for any old imports.
H2SpreadTableclothEnvCfg = G1SpreadTableclothEnvCfg

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

"""H2 + Sharpa Wave base environment configuration for the spread-tablecloth task.

This is the H2-specific base config.  It defines:
  - H2SpreadTableclothSceneCfg  (H2 robot, front camera only, no wrist cameras)
  - H2SpreadTableclothEnvCfg    (env-level MDP wiring)

The teleop variant (H2SpreadTableclothTeleopEnvCfg) inherits from this and
adds PinkIK actions + IsaacTeleop XR configuration.
"""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from isaaclab_physx.assets import DeformableObjectCfg
from isaaclab_physx.physics import PhysxCfg
from isaaclab_physx.sim import SurfaceDeformableBodyMaterialCfg
from simulation.tasks.spread_tablecloth import mdp

from simulation.tasks.spread_tablecloth.config import (  # isort: skip
    CameraPresets,
    H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER,
    H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    H2_SPREAD_TABLECLOTH_INIT_POS,
    H2_SPREAD_TABLECLOTH_INIT_ROT,
    H2RobotPresets,
)


TABLECLOTH_USD_PATH = (
    "/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/Assets/Cloth/Cloth_fold06/Cloth_fold13.usd"
)
SCENE_USD_PATH = (
    "/home/mxgu/Workspace/Omniverse/gmx/surgery-room-dev-internal/assets/Assets/scene04.usd"
)

# H2 body joints + Sharpa Wave hand joints (in PhysX BFS articulation order).
h2_joint_names = [
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
    "head_pitch_joint",
    "head_yaw_joint",
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
] + H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER


@configclass
class H2SpreadTableclothSceneCfg(InteractiveSceneCfg):
    """Scene configuration for the H2 spread-tablecloth task.

    H2 robot + head-mounted front camera only (no wrist cameras).
    """

    robot = H2RobotPresets.h2_sharpa_base_fix(
        init_pos=H2_SPREAD_TABLECLOTH_INIT_POS,
        init_rot=H2_SPREAD_TABLECLOTH_INIT_ROT,
        custom_joint_pos=H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    )

    front_camera = CameraPresets.h2_front_camera(focal_length=10.5)

    scene = AssetBaseCfg(
        prim_path="/World/envs/env_.*/Scene",
        spawn=sim_utils.UsdFileCfg(
            usd_path=SCENE_USD_PATH,
            scale=(1.0, 1.0, 1.3),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.9, -2.5, 0.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
    )

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tablecloth",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.50, 0.0, 1.0),
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLECLOTH_USD_PATH,
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
class H2ActionsCfg:
    """Direct joint angle control for H2 + Sharpa Wave hands."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=h2_joint_names,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class H2ObservationsCfg:
    """Observation configuration for the H2 spread-tablecloth task.

    Only front_camera — H2 has no wrist cameras.
    """

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
class H2TerminationsCfg:
    """Termination conditions — only time-out for now."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class H2EventCfg:
    """Event configuration for scene reset."""

    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")

    reset_cloth_inner = EventTermCfg(
        func=mdp.reset_cloth_inner,
        mode="reset",
        params={
            "cloth_asset_name": "cloth",
            "inner_rel_path": "Cloth_In002/Cloth_In002",
        },
    )


@configclass
class H2SpreadTableclothEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree H2 + Sharpa Wave spread-tablecloth environment configuration."""

    scene: H2SpreadTableclothSceneCfg = H2SpreadTableclothSceneCfg(
        num_envs=1,
        env_spacing=6.0,
        replicate_physics=False,
    )

    viewer: ViewerCfg = ViewerCfg(
        eye=(0.0, 0.8, 3.0),
        lookat=(-0.6, 0.5, 0.70),
        cam_prim_path="/OmniverseKit_Persp",
    )

    observations: H2ObservationsCfg = H2ObservationsCfg()
    actions: H2ActionsCfg = H2ActionsCfg()
    terminations: H2TerminationsCfg = H2TerminationsCfg()
    events: H2EventCfg = H2EventCfg()
    commands = None
    rewards = None
    curriculum = None

    def __post_init__(self):
        """Post initialization."""
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2
        self.sim.physics = PhysxCfg(
            bounce_threshold_velocity=0.01,
        )

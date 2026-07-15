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

"""H2 + Sharpa Wave env for the spread-tablecloth task."""

import isaaclab.envs.mdp as base_mdp
import isaaclab.sim as sim_utils
from isaaclab.assets import AssetBaseCfg, RigidObjectCfg
from isaaclab.assets.deformable_object import DeformableObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg, ViewerCfg
from isaaclab.managers import EventTermCfg
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.utils import configclass
from simulation.tasks.spread_tablecloth import CLOTH_INNER_USD, CLOTH_ONLY_USD, TABLE_USD, mdp
from simulation.tasks.spread_tablecloth.cloth_physics import make_newton_physics
from simulation.tasks.spread_tablecloth.config import (
    H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER,
    H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    H2_SPREAD_TABLECLOTH_INIT_POS,
    H2_SPREAD_TABLECLOTH_INIT_ROT,
    CameraPresets,
    H2RobotPresets,
)

# Body + Sharpa Wave hand joints in PhysX BFS articulation order.
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
    """H2 robot + head-mounted front camera (no wrist cameras)."""

    robot = H2RobotPresets.h2_sharpa_base_fix(
        init_pos=H2_SPREAD_TABLECLOTH_INIT_POS,
        init_rot=H2_SPREAD_TABLECLOTH_INIT_ROT,
        custom_joint_pos=H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS,
    )

    front_camera = CameraPresets.h2_front_camera(
        focal_length=10.5,
        pos_offset=(0.0, 0.0, -0.18),
    )

    # Ground plane at z=0.02 (the robot's feet rest here; do not move it).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(
            usd_path=TABLE_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(-0.42, 0.0, 0.385),  # tabletop top sits at z = 0.77 m
            rot=(0.0, 0.0, 0.70710678, 0.70710678),
        ),
    )

    cloth: DeformableObjectCfg = DeformableObjectCfg(
        prim_path="{ENV_REGEX_NS}/Tablecloth",
        init_state=DeformableObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.77),  # rest on the tabletop surface (z=0.77)
            rot=(0.0, 0.0, 0.0, 1.0),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=CLOTH_ONLY_USD,
        ),
    )

    cloth_inner = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/ClothInner",
        spawn=sim_utils.UsdFileCfg(
            usd_path=CLOTH_INNER_USD,
            rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True),
        ),
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(-0.55, 0.0, 0.83),  # cloth.pos.z + 0.06 (same offset as G1 config)
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
class H2ActionsCfg:
    """Direct joint angle control."""

    joint_pos = mdp.JointPositionActionCfg(
        asset_name="robot",
        joint_names=h2_joint_names,
        scale=1.0,
        use_default_offset=False,
        preserve_order=True,
    )


@configclass
class H2ObservationsCfg:
    """Joint state + front camera (H2 has no wrist cameras)."""

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
    """Time-out only."""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)


@configclass
class H2EventCfg:
    """Per-episode reset events. The inner rigid body (``cloth_inner``) is now
    a kinematic RigidObject so IsaacLab respawns it at ``init_state`` on reset
    automatically -- no explicit reset event needed."""

    reset_scene = EventTermCfg(func=base_mdp.reset_scene_to_default, mode="reset")


@configclass
class H2SpreadTableclothEnvCfg(ManagerBasedRLEnvCfg):
    """Unitree H2 + Sharpa Wave spread-tablecloth env."""

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
        self.decimation = 4
        self.episode_length_s = 30.0
        self.sim.dt = 1 / 120
        self.sim.render_interval = 2
        # Newton coupled MJWarp (robot) + VBD (cloth) backend.
        self.sim.physics = make_newton_physics()

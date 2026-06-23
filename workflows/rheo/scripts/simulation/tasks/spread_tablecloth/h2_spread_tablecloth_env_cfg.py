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

import os

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
# Table256 is 1.0 x 0.6 x 0.77 m with its origin at the bbox center (tabletop slab
# +0.385 m above origin). With pos.z = 0.385 the bottom sits at z=0 and the tabletop
# top is at z=0.77 m.
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

    front_camera = CameraPresets.h2_front_camera(focal_length=10.5)

    # Ground plane at z=0.02 (the robot's feet rest here; do not move it).
    ground = AssetBaseCfg(
        prim_path="/World/ground",
        spawn=sim_utils.GroundPlaneCfg(),
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.0, 0.0, 0.0),
        ),
    )

    # Table the cloth is spread on. The asset already ships its own colliders
    # (/root/Table256/Collisions/*), which Newton imports as fixed obstacles, so we
    # do not author extra collision. rot is a 90 deg yaw about +Z so the 1.0 m long
    # edge runs along Y. scale Z by 1.2727 makes the 0.77 m table 0.98 m tall; center
    # z=0.51 -> bottom on the ground (0.02), tabletop surface at z=1.0.
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        spawn=sim_utils.UsdFileCfg(usd_path=TABLE_USD),
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
            # Cloth-only composition wrapper. It (a) deactivates the embedded
            # rigid body so the deformable prim exposes exactly one mesh, and
            # (b) authors + binds the Newton surface-deformable material on the
            # template root itself. The cloth mesh in the original asset is
            # ALREADY an OmniPhysics surface deformable, so we must NOT pass
            # ``deformable_props`` (that would author a second simulation mesh),
            # and we must NOT pass ``physics_material`` (IsaacLab's
            # ``bind_physics_material`` refuses to bind on this wrapper Xform
            # root, which carries no physics API). See cloth_only.usda.
            usd_path=CLOTH_ONLY_USD,
        ),
    )

    # cloth_inner MUST be a RigidObjectCfg: the scene needs >=1 free rigid body or
    # Newton's finalize corrupts the heap ("double free") and aborts during sim init
    # (verified: AssetBaseCfg and removing it both crash this way).
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
    """Reset scene + reset the standalone inner rigid body back to its
    (cloth-wrapped) initial pose each episode."""

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
    
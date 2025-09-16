# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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


from dataclasses import MISSING
from typing import Any

import isaaclab.envs.mdp as mdp
import isaaclab.sim as sim_utils
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg, AssetBaseCfg, RigidObjectCfg
from isaaclab.envs import ManagerBasedRLEnvCfg
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg as RecordTerm
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.managers import ObservationGroupCfg as ObsGroup
from isaaclab.managers import ObservationTermCfg as ObsTerm
from isaaclab.managers import SceneEntityCfg
from isaaclab.managers import TerminationTermCfg as DoneTerm
from isaaclab.scene import InteractiveSceneCfg
from isaaclab.sensors import TiledCameraCfg
from isaaclab.utils import configclass
from leisaac.devices.action_process import init_action_cfg, preprocess_device_action
from simulation.utils.assets import so_arm_starter_assets


def reset_xform_root_pose_uniform(env, env_ids, pose_range, velocity_range, asset_cfg: SceneEntityCfg):
    """Reset an XForm prim's world pose around its initial pose using per-axis uniform offsets."""
    xform = env.scene[asset_cfg.name]  # XFormPrim
    positions, orientations = xform.get_world_poses()

    # Build indices tensor on correct device
    if hasattr(env_ids, "to"):
        idx_tensor = env_ids.to(dtype=torch.long, device=positions.device)
    else:
        idx_tensor = torch.tensor(env_ids, dtype=torch.long, device=positions.device)

    # Select current (base) poses — after reset_all, this is the initial pose
    pos_sel = positions.index_select(0, idx_tensor).clone()
    ori_sel = orientations.index_select(0, idx_tensor).clone()

    # Offsets sampled
    num_sel = idx_tensor.shape[0]
    offs = torch.zeros((num_sel, 3), device=positions.device, dtype=positions.dtype)
    for j, key in enumerate(["x", "y", "z"]):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        if lo != 0.0 or hi != 0.0:
            offs[:, j] = torch.empty(num_sel, device=positions.device, dtype=positions.dtype).uniform_(lo, hi)

    # Apply offsets relative to initial position
    base_pos = pos_sel[:, :3].clone()
    target_pos = base_pos + offs
    for j, key in enumerate(["x", "y", "z"]):
        lo, hi = pose_range.get(key, (0.0, 0.0))
        if lo != 0.0 or hi != 0.0:
            min_v = base_pos[:, j] + lo
            max_v = base_pos[:, j] + hi
            target_pos[:, j] = torch.minimum(torch.maximum(target_pos[:, j], min_v), max_v)

    pos_sel[:, :3] = target_pos

    xform.set_world_poses(pos_sel, ori_sel, indices=idx_tensor)


SOARM101_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=so_arm_starter_assets.SoArm,
        visible=True,
        copy_from_source=True,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
        ),
        collision_props=sim_utils.CollisionPropertiesCfg(
            contact_offset=0.005,
            rest_offset=0.001,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            solver_position_iteration_count=16,
            solver_velocity_iteration_count=4,
        ),
    ),
    init_state=ArticulationCfg.InitialStateCfg(
        # Position robot on top of the table surface
        pos=(0.4, 0.1, 0.2),
        rot=(0.707, 0.0, 0.0, -0.707),
        joint_pos={
            "shoulder_pan": 0.0,
            "shoulder_lift": 0.0,
            "elbow_flex": 0.0,
            "wrist_flex": 0.0,
            "wrist_roll": 0.0,
            "gripper": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Configure joint stiffness and damping for specific task control
        "arm_joints": ImplicitActuatorCfg(
            joint_names_expr=["shoulder_pan", "shoulder_lift", "elbow_flex", "wrist_flex", "wrist_roll"],
            effort_limit=5.2,
            velocity_limit=6.28,
            stiffness=80.0,
            damping=20.0,
        ),
        "gripper": ImplicitActuatorCfg(
            joint_names_expr=["gripper"],
            effort_limit=12.0,
            velocity_limit=31.4,
            stiffness=80.0,
            damping=10.0,
        ),
    },
)


@configclass
class SoArm101TableSceneCfg(InteractiveSceneCfg):
    """Configuration for SO-ARM 101 with table environment and camera sensors for recording."""

    # Ground plane
    ground = AssetBaseCfg(
        prim_path="/World/GroundPlane",
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, -1.05)),
        spawn=sim_utils.GroundPlaneCfg(),
    )

    robot = SOARM101_CFG.replace(prim_path="{ENV_REGEX_NS}/robot")

    wrist = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/robot/gripper/visuals/pcb_board_36x36/Camera",
        spawn=None,
        data_types=["rgb"],
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    room: TiledCameraCfg = TiledCameraCfg(
        prim_path="{ENV_REGEX_NS}/RoomCamera",
        offset=TiledCameraCfg.OffsetCfg(pos=(0.12, 0.08, 0.7), rot=(0.0, 0.7071, -0.7071, 0.0), convention="ros"),
        data_types=["rgb"],
        spawn=sim_utils.PinholeCameraCfg(
            focal_length=16.0,
            focus_distance=100.0,
            horizontal_aperture=20.955,
            clipping_range=(0.1, 1.0e5),
        ),
        width=640,
        height=480,
        update_period=1 / 30.0,  # 30FPS
    )

    # Table - Seattle Lab Table from Isaac Nucleus
    table = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Table",
        init_state=AssetBaseCfg.InitialStateCfg(
            pos=(0.1, 0.0, 0.0),  # Table at origin
            rot=(0.707, 0.0, 0.0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=so_arm_starter_assets.Table,
            copy_from_source=True,
            visible=True,
            scale=(0.7, 0.7, 0.52),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.001,
            ),
        ),
    )

    scissors = RigidObjectCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalScissors",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.12, -0.02, 0.0),
            rot=(0.707, 0, 0, 0.707),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=so_arm_starter_assets.Scissors,
            scale=(0.006, 0.0065, 0.012),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            collision_props=sim_utils.CollisionPropertiesCfg(
                contact_offset=0.005,
                rest_offset=0.001,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=0.15,
            ),
        ),
    )

    tray = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/SurgicalTray",
        init_state=RigidObjectCfg.InitialStateCfg(
            pos=(0.12, 0.25, 0.26),
            rot=(0.7071, 0.0, 0.0, 0.7071),
        ),
        spawn=sim_utils.UsdFileCfg(
            usd_path=so_arm_starter_assets.Tray,
            scale=(0.7, 0.7, 0.18),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.5, 0.5, 0.5),  # Silver appearance
                metallic=0.8,
                roughness=0.25,
            ),
            rigid_props=sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
            ),
            mass_props=sim_utils.MassPropertiesCfg(
                mass=5.0,
            ),
        ),
    )

    # Dome light for proper lighting
    dome_light = AssetBaseCfg(
        prim_path="/World/DomeLight",
        spawn=sim_utils.DomeLightCfg(
            intensity=2200.0,
            color=(0.8, 0.8, 0.8),  #  softer color
        ),
    )

    # Additional directional light for better robot visibility
    directional_light = AssetBaseCfg(
        prim_path="/World/DirectionalLight",
        spawn=sim_utils.DistantLightCfg(
            intensity=800.0,
            color=(0.95, 0.95, 0.9),  # Slightly warmer, softer white
            angle=45.0,
        ),
    )


@configclass
class ActionsCfg:
    """Configuration for the actions."""

    arm_action: mdp.ActionTermCfg = MISSING
    gripper_action: mdp.ActionTermCfg = MISSING


@configclass
class EventCfg:
    """Configuration for the events."""

    robot_physics_material = EventTerm(
        func=mdp.randomize_rigid_body_material,
        mode="startup",
        params={
            "asset_cfg": SceneEntityCfg("robot", body_names=".*"),
            "static_friction_range": (0.8, 1.25),
            "dynamic_friction_range": (0.8, 1.25),
            "restitution_range": (0.0, 0.0),
            "num_buckets": 16,
        },
    )
    # reset to default scene
    reset_all = EventTerm(func=mdp.reset_scene_to_default, mode="reset")

    # Reset scissors with small randomization
    reset_scissors = EventTerm(
        func=mdp.reset_root_state_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.01, 0.010),
                "y": (-0.005, 0.010),
                "z": (0.0, 0.0),
                "roll": (-0.0, 0.0),
                "pitch": (-0.0, 0.0),
                "yaw": (-0.2, 0.2),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("scissors"),
        },
    )

    # Reset tray with small randomization (for AssetBaseCfg XForm prim)
    reset_tray = EventTerm(
        func=reset_xform_root_pose_uniform,
        mode="reset",
        params={
            "pose_range": {
                "x": (-0.005, 0.005),
                "y": (-0.005, 0.005),
                "z": (-0.000, 0.000),
            },
            "velocity_range": {},
            "asset_cfg": SceneEntityCfg("tray"),
        },
    )


@configclass
class ObservationsCfg:
    """Observation specifications for the MDP."""

    @configclass
    class PolicyCfg(ObsGroup):
        """Observations for policy group."""

        joint_pos = ObsTerm(func=mdp.joint_pos)
        joint_vel = ObsTerm(func=mdp.joint_vel)
        joint_pos_rel = ObsTerm(func=mdp.joint_pos_rel)
        joint_vel_rel = ObsTerm(func=mdp.joint_vel_rel)
        actions = ObsTerm(func=mdp.last_action)
        wrist = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("wrist"), "data_type": "rgb", "normalize": False}
        )
        room = ObsTerm(
            func=mdp.image, params={"sensor_cfg": SceneEntityCfg("room"), "data_type": "rgb", "normalize": False}
        )

        def __post_init__(self):
            self.enable_corruption = True
            self.concatenate_terms = False

    # observation groups
    policy: PolicyCfg = PolicyCfg()


@configclass
class RewardsCfg:
    """Configuration for the rewards"""


@configclass
class TerminationsCfg:
    """Configuration for the termination"""

    time_out = DoneTerm(func=mdp.time_out, time_out=True)

    # success = DoneTerm(func=mdp.task_done, params={
    #     "scissors_cfg": SceneEntityCfg("scissors"),
    #     "tray_cfg": SceneEntityCfg("tray")
    # })
    # Note: Using manual success termination via 'N' key in teleoperation instead


@configclass
class SOARMStarterEnvCfg(ManagerBasedRLEnvCfg):
    """Configuration for the SO-ARM Starter environment."""

    scene: SoArm101TableSceneCfg = SoArm101TableSceneCfg(env_spacing=4.0)

    observations: ObservationsCfg = ObservationsCfg()
    actions: ActionsCfg = ActionsCfg()
    events: EventCfg = EventCfg()

    rewards: RewardsCfg = RewardsCfg()
    terminations: TerminationsCfg = TerminationsCfg()

    recorders: RecordTerm = RecordTerm()

    def __post_init__(self) -> None:
        super().__post_init__()

        self.decimation = 1
        self.episode_length_s = 8.0
        self.viewer.eye = (2.0, 2.0, 1.5)
        self.viewer.lookat = (0.0, 0.0, 0.2)
        self.actions = init_action_cfg(self.actions, device="keyboard")

        self.sim.physx.bounce_threshold_velocity = 0.01
        self.sim.physx.friction_correlation_distance = 0.00625
        self.sim.render.enable_translucency = True

    def use_teleop_device(self, teleop_device) -> None:
        self.actions = init_action_cfg(self.actions, device=teleop_device)
        if teleop_device == "keyboard":
            self.scene.robot.spawn.rigid_props.disable_gravity = True

    def preprocess_device_action(self, action: dict[str, Any], teleop_device) -> torch.Tensor:
        return preprocess_device_action(action, teleop_device)

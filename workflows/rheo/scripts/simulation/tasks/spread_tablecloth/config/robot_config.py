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

"""Robot configs for spread_tablecloth (G1 29DOF + Inspire / H2 + Sharpa Wave)."""

import os
from typing import Dict, Optional, Tuple

import isaaclab.sim as sim_utils
from isaaclab.actuators import IdealPDActuatorCfg, ImplicitActuatorCfg
from isaaclab.assets import ArticulationCfg
from isaaclab.utils import configclass

_ASSET_ROOT = "https://omniverse-content-production.s3-us-west-2.amazonaws.com/Assets/Isaac/Healthcare/0.7.0/724f82e"
G1_INSPIRE_USD_PATH = f"{_ASSET_ROOT}/Robots/UnitreeG1/g1_29dof_with_inspire_rev_1_0/g1_29dof_with_inspire_rev_1_0.usd"
H2_SHARPA_USD_PATH = f"{_ASSET_ROOT}/Robots/UnitreeH2/h2_with_sharpa/H2_with_sharpa_flat.usd"
# ---------------------------------------------------------------------------
# H2 + Sharpa non-USD asset resolution (urdf/ + teleop_configs/)
# ---------------------------------------------------------------------------
_H2_SHARPA_ASSET_ROOT = os.path.expanduser(
    os.environ.get(
        "RHEO_H2_SHARPA_ASSETS_DIR",
        "~/.cache/i4h_workflows/spread_tablecloth/h2_with_sharpa",
    )
)
H2_SHARPA_URDF_PATH = os.path.join(_H2_SHARPA_ASSET_ROOT, "urdf", "H2_with_sharpa_hands.urdf")
H2_SHARPA_HAND_URDF_DIR = os.path.join(_H2_SHARPA_ASSET_ROOT, "urdf", "sharpa_standalone")
H2_SHARPA_TELEOP_CONFIG_DIR = os.path.join(_H2_SHARPA_ASSET_ROOT, "teleop_configs")

_H2_SHARPA_SUBDIRS = (
    ("urdf", "H2_with_sharpa_hands.urdf"),
    ("teleop_configs", "sharpa_wave_left_dexpilot.yml"),
)


def ensure_h2_sharpa_assets(usd_url: str = H2_SHARPA_USD_PATH) -> None:
    """Mirror h2_with_sharpa/urdf and teleop_configs from Nucleus into the
    local cache. No-op when the marker files already exist."""
    if not usd_url.startswith(("omniverse://", "http://", "https://")):
        raise FileNotFoundError(
            f"H2_SHARPA_USD_PATH ({usd_url!r}) is not a Nucleus URL; set "
            "RHEO_H2_SHARPA_ASSETS_DIR to an already-populated local folder "
            "or point H2_SHARPA_USD_PATH at Nucleus."
        )
    base_url = usd_url.rsplit("/", 1)[0] + "/"
    for subdir, marker in _H2_SHARPA_SUBDIRS:
        local = os.path.join(_H2_SHARPA_ASSET_ROOT, subdir)
        if os.path.isfile(os.path.join(local, marker)):
            continue
        os.makedirs(local, exist_ok=True)
        import omni.client  # noqa: PLC0415  (deferred; skipped when cache is warm)

        remote = base_url + subdir + "/"
        print(f"[rheo] downloading {remote} -> {local}", flush=True)
        result = omni.client.copy(remote, local, omni.client.CopyBehavior.OVERWRITE)
        if result != omni.client.Result.OK:
            raise RuntimeError(f"omni.client.copy {remote} -> {result}")


# ---------------------------------------------------------------------------
# H2 + Sharpa Wave hands
# ---------------------------------------------------------------------------
H2_SHARPA_HAND_JOINT_NAMES_ARTICULATION_ORDER = [
    "left_index_MCP_FE",
    "left_middle_MCP_FE",
    "left_pinky_CMC",
    "left_ring_MCP_FE",
    "left_thumb_CMC_FE",
    "right_index_MCP_FE",
    "right_middle_MCP_FE",
    "right_pinky_CMC",
    "right_ring_MCP_FE",
    "right_thumb_CMC_FE",
    "left_index_MCP_AA",
    "left_middle_MCP_AA",
    "left_pinky_MCP_FE",
    "left_ring_MCP_AA",
    "left_thumb_CMC_AA",
    "right_index_MCP_AA",
    "right_middle_MCP_AA",
    "right_pinky_MCP_FE",
    "right_ring_MCP_AA",
    "right_thumb_CMC_AA",
    "left_index_PIP",
    "left_middle_PIP",
    "left_pinky_MCP_AA",
    "left_ring_PIP",
    "left_thumb_MCP_FE",
    "right_index_PIP",
    "right_middle_PIP",
    "right_pinky_MCP_AA",
    "right_ring_PIP",
    "right_thumb_MCP_FE",
    "left_index_DIP",
    "left_middle_DIP",
    "left_pinky_PIP",
    "left_ring_DIP",
    "left_thumb_MCP_AA",
    "right_index_DIP",
    "right_middle_DIP",
    "right_pinky_PIP",
    "right_ring_DIP",
    "right_thumb_MCP_AA",
    "left_pinky_DIP",
    "left_thumb_IP",
    "right_pinky_DIP",
    "right_thumb_IP",
]

H2_DEFAULT_JOINT_POS: Dict[str, float] = {
    "left_hip_pitch_joint": -0.1,
    "left_hip_roll_joint": 0.0,
    "left_hip_yaw_joint": 0.0,
    "left_knee_joint": 0.3,
    "left_ankle_roll_joint": 0.0,
    "left_ankle_pitch_joint": -0.2,
    "right_hip_pitch_joint": -0.1,
    "right_hip_roll_joint": 0.0,
    "right_hip_yaw_joint": 0.0,
    "right_knee_joint": 0.3,
    "right_ankle_roll_joint": 0.0,
    "right_ankle_pitch_joint": -0.2,
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    "head_pitch_joint": 0.0,
    "head_yaw_joint": 0.0,
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": 0.0,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": 0.0,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}

# z=1.05 lifts the feet to the ground; the H2_Plus USD has no built-in offset.
H2_SPREAD_TABLECLOTH_INIT_POS: Tuple[float, float, float] = (-0.95, 0.0, 1.05)
H2_SPREAD_TABLECLOTH_INIT_ROT: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
H2_SPREAD_TABLECLOTH_CUSTOM_JOINT_POS: Dict[str, float] = {
    "left_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.5,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": -0.5,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


H2_SHARPA_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=H2_SHARPA_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            fix_root_link=True,
            solver_position_iteration_count=4,
            solver_velocity_iteration_count=0,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=H2_SPREAD_TABLECLOTH_INIT_POS,
        rot=H2_SPREAD_TABLECLOTH_INIT_ROT,
        joint_pos=H2_DEFAULT_JOINT_POS,
        joint_vel={".*": 0.0},
    ),
    actuators={
        # Legs/feet are not commanded in this upper-body teleop task.
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=1000.0,
            armature=0.03,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_roll_joint", ".*_ankle_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness=10000.0,
            damping=1000.0,
            armature=0.03,
        ),
        "waist": IdealPDActuatorCfg(
            joint_names_expr=["waist_.*_joint"],
            effort_limit={
                "waist_yaw_joint": 120.0,
                "waist_roll_joint": 180.0,
                "waist_pitch_joint": 180.0,
            },
            velocity_limit=28.0,
            stiffness=1e4,
            damping=1e3,
            armature=0.03,
            friction=0.03,
        ),
        "head": IdealPDActuatorCfg(
            joint_names_expr=["head_.*_joint"],
            effort_limit=50.0,
            velocity_limit=10.0,
            stiffness=40.0,
            damping=2.0,
            armature=0.03,
            friction=0.03,
        ),
        "arms": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_pitch_joint",
                ".*_shoulder_roll_joint",
                ".*_shoulder_yaw_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit={
                ".*_shoulder_pitch_joint": 220.0,
                ".*_shoulder_roll_joint": 154.0,
                ".*_shoulder_yaw_joint": 154.0,
                ".*_elbow_joint": 154.0,
                ".*_wrist_roll_joint": 154.0,
                ".*_wrist_pitch_joint": 125.0,
                ".*_wrist_yaw_joint": 125.0,
            },
            velocity_limit={
                ".*_shoulder_pitch_joint": 28.0,
                ".*_shoulder_roll_joint": 34.0,
                ".*_shoulder_yaw_joint": 34.0,
                ".*_elbow_joint": 34.0,
                ".*_wrist_roll_joint": 34.0,
                ".*_wrist_pitch_joint": 50.0,
                ".*_wrist_yaw_joint": 50.0,
            },
            stiffness={
                ".*_shoulder_pitch_joint": 200.0,
                ".*_shoulder_roll_joint": 200.0,
                ".*_shoulder_yaw_joint": 140.0,
                ".*_elbow_joint": 140.0,
                ".*_wrist_.*_joint": 200.0,
            },
            damping={
                ".*_shoulder_pitch_joint": 5.0,
                ".*_shoulder_roll_joint": 5.0,
                ".*_shoulder_yaw_joint": 2.0,
                ".*_elbow_joint": 2.0,
                ".*_wrist_.*_joint": 5.0,
            },
            armature={".*_shoulder_.*": 0.03, ".*_elbow_.*": 0.03, ".*_wrist_.*_joint": 0.03},
            friction=0.03,
        ),
        "hands": IdealPDActuatorCfg(
            joint_names_expr=[
                ".*_thumb_.*",
                ".*_index_.*",
                ".*_middle_.*",
                ".*_ring_.*",
                ".*_pinky_.*",
            ],
            effort_limit=3.3,
            velocity_limit=16.0,
            stiffness=4.0,
            damping=0.5,
            armature=0.03,
            friction=0.03,
        ),
    },
)


def make_h2_sharpa_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: Tuple[float, float, float] = H2_SPREAD_TABLECLOTH_INIT_POS,
    init_rot: Tuple[float, float, float, float] = H2_SPREAD_TABLECLOTH_INIT_ROT,
    custom_joint_pos: Optional[Dict[str, float]] = None,
    base_config: ArticulationCfg = H2_SHARPA_CFG,
) -> ArticulationCfg:
    joint_pos = H2_DEFAULT_JOINT_POS.copy()
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class H2RobotPresets:
    """H2 robot presets."""

    @classmethod
    def h2_sharpa_base_fix(
        cls,
        init_pos: Tuple[float, float, float] = H2_SPREAD_TABLECLOTH_INIT_POS,
        init_rot: Tuple[float, float, float, float] = H2_SPREAD_TABLECLOTH_INIT_ROT,
        custom_joint_pos: Optional[Dict[str, float]] = None,
    ) -> ArticulationCfg:
        """H2 + Sharpa Wave, base-fixed."""
        return make_h2_sharpa_cfg(
            init_pos=init_pos,
            init_rot=init_rot,
            custom_joint_pos=custom_joint_pos,
        )


# ---------------------------------------------------------------------------
# G1 + Inspire hands
# ---------------------------------------------------------------------------

DEFAULT_JOINT_POS: Dict[str, float] = {
    # legs
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
    # waist
    "waist_yaw_joint": 0.0,
    "waist_roll_joint": 0.0,
    "waist_pitch_joint": 0.0,
    # arms
    "left_shoulder_pitch_joint": 0.0,
    "left_shoulder_roll_joint": 0.0,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.3,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": 0.0,
    "right_shoulder_roll_joint": 0.0,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.3,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
    # inspire hands (left)
    "L_index_proximal_joint": 0.0,
    "L_index_intermediate_joint": 0.0,
    "L_middle_proximal_joint": 0.0,
    "L_middle_intermediate_joint": 0.0,
    "L_pinky_proximal_joint": 0.0,
    "L_pinky_intermediate_joint": 0.0,
    "L_ring_proximal_joint": 0.0,
    "L_ring_intermediate_joint": 0.0,
    "L_thumb_proximal_yaw_joint": 0.0,
    "L_thumb_proximal_pitch_joint": 0.0,
    "L_thumb_intermediate_joint": 0.0,
    "L_thumb_distal_joint": 0.0,
    # inspire hands (right)
    "R_index_proximal_joint": 0.0,
    "R_index_intermediate_joint": 0.0,
    "R_middle_proximal_joint": 0.0,
    "R_middle_intermediate_joint": 0.0,
    "R_pinky_proximal_joint": 0.0,
    "R_pinky_intermediate_joint": 0.0,
    "R_ring_proximal_joint": 0.0,
    "R_ring_intermediate_joint": 0.0,
    "R_thumb_proximal_yaw_joint": 0.0,
    "R_thumb_proximal_pitch_joint": 0.0,
    "R_thumb_intermediate_joint": 0.0,
    "R_thumb_distal_joint": 0.0,
}

SPREAD_TABLECLOTH_INIT_POS: Tuple[float, float, float] = (-0.95, 0.0, 0.80)
SPREAD_TABLECLOTH_INIT_ROT: Tuple[float, float, float, float] = (0.0, 0.0, 0.0, 1.0)
SPREAD_TABLECLOTH_CUSTOM_JOINT_POS: Dict[str, float] = {
    "left_shoulder_pitch_joint": -0.3,
    "left_shoulder_roll_joint": 0.5,
    "left_shoulder_yaw_joint": 0.0,
    "left_elbow_joint": -0.5,
    "left_wrist_roll_joint": 0.0,
    "left_wrist_pitch_joint": 0.0,
    "left_wrist_yaw_joint": 0.0,
    "right_shoulder_pitch_joint": -0.3,
    "right_shoulder_roll_joint": -0.5,
    "right_shoulder_yaw_joint": 0.0,
    "right_elbow_joint": -0.5,
    "right_wrist_roll_joint": 0.0,
    "right_wrist_pitch_joint": 0.0,
    "right_wrist_yaw_joint": 0.0,
}


G129_CFG_WITH_INSPIRE_HAND = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        usd_path=G1_INSPIRE_USD_PATH,
        activate_contact_sensors=True,
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # Fixed-base manipulation robot: disable gravity on the robot bodies
            # so the arm does not sag (matches official G1_INSPIRE_FTP_CFG). World
            # gravity stays on, so the cloth still drapes.
            disable_gravity=True,
            retain_accelerations=False,
            linear_damping=0.0,
            angular_damping=0.0,
            max_linear_velocity=1000.0,
            max_angular_velocity=1000.0,
            max_depenetration_velocity=1.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=False,
            # True fixed base. On Newton/MJWarp a floating root plus the USD
            # fixed-to-world root_joint over-constrains the base and diverges to
            # NaN on the first actuated step; fixing the root link avoids this
            # (matches official G1_INSPIRE_FTP_CFG).
            fix_root_link=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        ),
    ),
    prim_path="/World/envs/env_.*/Robot",
    init_state=ArticulationCfg.InitialStateCfg(
        pos=(0.0, 0.0, 0.75),
        joint_pos={
            "left_hip_yaw_joint": 0.0,
            "left_hip_roll_joint": 0.0,
            "left_hip_pitch_joint": -0.05,
            "left_knee_joint": 0.2,
            "left_ankle_pitch_joint": -0.15,
            "left_ankle_roll_joint": 0.0,
            "right_hip_yaw_joint": 0.0,
            "right_hip_roll_joint": 0.0,
            "right_hip_pitch_joint": -0.05,
            "right_knee_joint": 0.2,
            "right_ankle_pitch_joint": -0.15,
            "right_ankle_roll_joint": 0.0,
            "waist_yaw_joint": 0.0,
            "waist_roll_joint": 0.0,
            "waist_pitch_joint": 0.0,
            "left_shoulder_pitch_joint": 0.0,
            "left_shoulder_roll_joint": 0.0,
            "left_shoulder_yaw_joint": 0.0,
            "left_elbow_joint": 0.0,
            "left_wrist_roll_joint": 0.0,
            "left_wrist_pitch_joint": 0.0,
            "left_wrist_yaw_joint": 0.0,
            "right_shoulder_pitch_joint": 0.0,
            "right_shoulder_roll_joint": 0.0,
            "right_shoulder_yaw_joint": 0.0,
            "right_elbow_joint": 0.0,
            "right_wrist_roll_joint": 0.0,
            "right_wrist_pitch_joint": 0.0,
            "right_wrist_yaw_joint": 0.0,
            "L_index_proximal_joint": 0.0,
            "L_index_intermediate_joint": 0.0,
            "L_middle_proximal_joint": 0.0,
            "L_middle_intermediate_joint": 0.0,
            "L_pinky_proximal_joint": 0.0,
            "L_pinky_intermediate_joint": 0.0,
            "L_ring_proximal_joint": 0.0,
            "L_ring_intermediate_joint": 0.0,
            "L_thumb_proximal_yaw_joint": 0.0,
            "L_thumb_proximal_pitch_joint": 0.0,
            "L_thumb_intermediate_joint": 0.0,
            "L_thumb_distal_joint": 0.0,
            "R_index_proximal_joint": 0.0,
            "R_index_intermediate_joint": 0.0,
            "R_middle_proximal_joint": 0.0,
            "R_middle_intermediate_joint": 0.0,
            "R_pinky_proximal_joint": 0.0,
            "R_pinky_intermediate_joint": 0.0,
            "R_ring_proximal_joint": 0.0,
            "R_ring_intermediate_joint": 0.0,
            "R_thumb_proximal_yaw_joint": 0.0,
            "R_thumb_proximal_pitch_joint": 0.0,
            "R_thumb_intermediate_joint": 0.0,
            "R_thumb_distal_joint": 0.0,
        },
        joint_vel={".*": 0.0},
    ),
    soft_joint_pos_limit_factor=0.9,
    actuators={
        "legs": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_hip_yaw_joint",
                ".*_hip_roll_joint",
                ".*_hip_pitch_joint",
                ".*_knee_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            stiffness=None,
            damping=None,
            armature=None,
        ),
        "waist": ImplicitActuatorCfg(
            joint_names_expr=["waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint"],
            effort_limit=1000.0,
            velocity_limit=0.0,
            stiffness={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
            damping={"waist_yaw_joint": 10000.0, "waist_roll_joint": 10000.0, "waist_pitch_joint": 10000.0},
            armature=None,
        ),
        "feet": ImplicitActuatorCfg(
            joint_names_expr=[".*_ankle_pitch_joint", ".*_ankle_roll_joint"],
            effort_limit=None,
            stiffness=None,
            damping=None,
        ),
        "arms": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_shoulder_.*_joint",
                ".*_elbow_joint",
                ".*_wrist_.*_joint",
            ],
            effort_limit=None,
            velocity_limit=None,
            # Was shoulder=25, elbow=50, wrist=40 with damping=2 across the
            # board -> natural frequency ~1.9 Hz on the arm inertia which is
            # slower than the ~30 Hz teleop update rate, giving the "arm
            # lags then catches up in a burst" feel the user reports (H2 vs G1
            # comparison: G1 arms use IdealPD kp=200/140 -> ~5.5 Hz -> tracks
            # smoothly). Match G1's kp band and up the damping proportionally
            # so we get its snappier response without introducing overshoot.
            stiffness={
                ".*_shoulder_.*_joint": 200.0,
                ".*_elbow_joint": 140.0,
                ".*_wrist_.*_joint": 60.0,
            },
            damping={
                ".*_shoulder_.*_joint": 20.0,
                ".*_elbow_joint": 15.0,
                ".*_wrist_.*_joint": 8.0,
            },
            armature=None,
        ),
        "hands": ImplicitActuatorCfg(
            joint_names_expr=[
                ".*_index_proximal_joint",
                ".*_index_intermediate_joint",
                ".*_middle_proximal_joint",
                ".*_middle_intermediate_joint",
                ".*_pinky_proximal_joint",
                ".*_pinky_intermediate_joint",
                ".*_ring_proximal_joint",
                ".*_ring_intermediate_joint",
                ".*_thumb_proximal_yaw_joint",
                ".*_thumb_proximal_pitch_joint",
                ".*_thumb_intermediate_joint",
                ".*_thumb_distal_joint",
            ],
            effort_limit=100.0,
            velocity_limit=50.0,
            stiffness={
                ".*_index_proximal_joint": 1000.0,
                ".*_index_intermediate_joint": 1000.0,
                ".*_middle_proximal_joint": 1000.0,
                ".*_middle_intermediate_joint": 1000.0,
                ".*_pinky_proximal_joint": 1000.0,
                ".*_pinky_intermediate_joint": 1000.0,
                ".*_ring_proximal_joint": 1000.0,
                ".*_ring_intermediate_joint": 1000.0,
                ".*_thumb_proximal_yaw_joint": 1000.0,
                ".*_thumb_proximal_pitch_joint": 1000.0,
                ".*_thumb_intermediate_joint": 1000.0,
                ".*_thumb_distal_joint": 1000.0,
            },
            damping={
                ".*_index_proximal_joint": 15.0,
                ".*_index_intermediate_joint": 15.0,
                ".*_middle_proximal_joint": 15.0,
                ".*_middle_intermediate_joint": 15.0,
                ".*_pinky_proximal_joint": 15.0,
                ".*_pinky_intermediate_joint": 15.0,
                ".*_ring_proximal_joint": 15.0,
                ".*_ring_intermediate_joint": 15.0,
                ".*_thumb_proximal_yaw_joint": 15.0,
                ".*_thumb_proximal_pitch_joint": 15.0,
                ".*_thumb_intermediate_joint": 15.0,
                ".*_thumb_distal_joint": 15.0,
            },
            armature={".*": 0.0},
        ),
    },
)


def make_g1_29dof_inspire_cfg(
    *,
    prim_path: str = "/World/envs/env_.*/Robot",
    init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.76),
    init_rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.7071, 0.7071),
    custom_joint_pos: Optional[Dict[str, float]] = None,
    base_config: ArticulationCfg = G129_CFG_WITH_INSPIRE_HAND,
) -> ArticulationCfg:
    joint_pos = DEFAULT_JOINT_POS.copy()
    if custom_joint_pos:
        joint_pos.update(custom_joint_pos)
    return base_config.replace(
        prim_path=prim_path,
        init_state=ArticulationCfg.InitialStateCfg(
            pos=init_pos,
            rot=init_rot,
            joint_pos=joint_pos,
            joint_vel={".*": 0.0},
        ),
    )


@configclass
class G1RobotPresets:
    """G1 robot presets."""

    @classmethod
    def g1_29dof_inspire_base_fix(
        cls,
        init_pos: Tuple[float, float, float] = (-0.15, 0.0, 0.76),
        init_rot: Tuple[float, float, float, float] = (0.0, 0.0, 0.7071, 0.7071),
        custom_joint_pos: Optional[Dict[str, float]] = None,
    ) -> ArticulationCfg:
        """G1 29DOF + Inspire hands, base-fixed."""
        return make_g1_29dof_inspire_cfg(
            init_pos=init_pos,
            init_rot=init_rot,
            custom_joint_pos=custom_joint_pos,
        )

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""dVRK PSM settings for the Newton (MuJoCo Warp) physics preset.

The PSM's drives and its lift workflows were tuned against PhysX, and three
PhysX behaviours they relied on do not carry over (NVBug 6928490):

- ``disable_gravity`` is a ``physxRigidBody`` attribute. Newton reads gravity
  per scene, not per body, so under Newton the arm hung under gravity: the
  insertion joint sagged 7-8 mm off its target and every Cartesian target was
  missed by that much. MuJoCo's per-body gravity compensation restores the
  intended weightless arm.
- PhysX integrates drives implicitly, so a 20 g jaw driven at stiffness 500
  with a 0.1 N m limit is stable there. MuJoCo integrates stiffness explicitly:
  the jaws oscillated past their +-0.52 rad limits from the first step, and
  the saturated drive closed them straight through the object. Armature gives
  the light tool links the inertia an explicit step needs, and a soft,
  critically damped jaw drive closes onto the object without overshooting.
- PhysX registers contact slightly before surfaces meet, so the lift
  workflows close the jaws with the tool tip at the object's origin. Newton's
  contacts are exact, and there the jaws only bite 5 mm further onto the
  object; the same 5 mm deeper fails every PhysX grasp, so the offset is
  Newton's alone.

Light on purpose: nothing here imports IsaacLab at module scope, so the
values can be checked without Kit.
"""

from __future__ import annotations

from typing import Any

#: Joint armature added to the six arm joints [kg m^2, kg for insertion].
ARM_ARMATURE = 0.01
#: Joint armature added to each jaw [kg m^2].
JAW_ARMATURE = 0.002
#: Jaw drive: critically damped for ``JAW_ARMATURE`` (2 * sqrt(k * I) = 0.126),
#: so the jaws stop where they meet the object instead of swinging past it.
#: PhysX's 500 / 0.1 drive with a 0.1 N m limit was saturated for the whole
#: close and had no braking left at the end.
JAW_STIFFNESS = 2.0
JAW_DAMPING = 0.12
JAW_EFFORT_LIMIT = 0.5
#: How much further onto the object the jaws close under Newton [m], along the
#: tool, which points straight down the PSM root's -z at the grasp.
GRASP_DEPTH_M = 0.005
#: Jaw close target for the suture needle under Newton [rad]. A wire carries no
#: squeeze at PhysX's 0.09 with this soft drive; closing fully keeps
#: the needle pinched with the drive's own stiffness.
NEEDLE_GRIPPER_CLOSE = 0.0


def configure_psm_articulation(robot_cfg: Any) -> None:
    """Apply the Newton settings above to one PSM ``ArticulationCfg``, in place."""
    from isaaclab_newton.sim.schemas.schemas_cfg import MujocoRigidBodyPropertiesCfg  # noqa: PLC0415

    robot_cfg.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.0, disable_gravity=True)
    robot_cfg.actuators["psm"].armature = ARM_ARMATURE
    jaw = robot_cfg.actuators["psm_tool"]
    jaw.armature = JAW_ARMATURE
    jaw.stiffness = JAW_STIFFNESS
    jaw.damping = JAW_DAMPING
    jaw.effort_limit_sim = JAW_EFFORT_LIMIT


def compensate_gravity_if_disabled(asset_cfg: Any) -> None:
    """Give a rigid object that PhysX holds weightless the same treatment under Newton."""
    from isaaclab_newton.sim.schemas.schemas_cfg import MujocoRigidBodyPropertiesCfg  # noqa: PLC0415

    spawn = getattr(asset_cfg, "spawn", None)
    rigid_props = getattr(spawn, "rigid_props", None)
    if rigid_props is not None and getattr(rigid_props, "disable_gravity", False):
        spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.0, disable_gravity=True)

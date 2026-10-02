# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Catheter embodiment whose drive unit rides on a Franka Panda flange.

A clinical endovascular robot advances the wire through a drive unit bolted to
an arm, so the arm belongs inside the control loop rather than beside it. The
policy still asks for insertion and rotation, but the request goes to the arm
first: the flange is servo'd along the introducer axis, MJWarp integrates the
arm, and the catheter is fed by the pose the flange *actually reached*. An arm
that is lagging, at a joint limit, or held by contact therefore stops feeding
wire, which a directly-driven catheter could never express.

Adding the arm also changes the physics backend. The rod solver integrates
particles and has no rigid integrator, so an articulation in the model means the
scene needs MJWarp beside the rod; that is what ``rigid_bodies_enabled`` on the
rod spec selects.

The action space is deliberately unchanged at insertion, rotation, and C-arm
orbit. The arm is servo'd rather than commanded, so it contributes joints to the
recorded state but no new action dimensions, and datasets keep the column layout
the plain catheter already established.

The articulation config is built here rather than imported from
``i4h_arena.embodiments.franka``. That module describes its arms with
PhysX-specific schemas while this scene runs Newton, its ultrasound variants
bring cameras and observation groups this scene has no use for, and its
``FRANKA_PANDA_CFG`` names an Isaac Lab nucleus path that no longer resolves.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.actuators import ImplicitActuatorCfg
from isaaclab.assets import AssetBaseCfg
from isaaclab.assets.articulation import ArticulationCfg
from isaaclab.controllers import DifferentialIKController, DifferentialIKControllerCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils import math as math_utils
from isaaclab.utils.configclass import configclass

from i4h_arena.assets.constants import FRANKA_PANDA_HAND_USD
from i4h_arena.embodiments.catheter import CArmOrbitAction, CArmOrbitActionCfg, CatheterEmbodiment, _CatheterSceneCfg
from i4h_arena.medical.catheter_drive import (
    FlangeMountedIntroducer,
    IntroducerDriveSpec,
    LumenClamp,
)
from i4h_arena.medical.newton_catheter_physics import newton_physics_cfg
from i4h_arena.medical.xpbd_catheter import XpbdCatheterAsset
from i4h_common.types import JointState

#: Isaac Lab's stock Franka, carrying the two-finger hand.
#:
#: The healthcare bundle's Panda was used here first, because the nucleus path
#: ``embodiments.franka`` names for ``FRANKA_PANDA_CFG`` 404s against the 6.0
#: asset layout. That bundle's arm ends at a bare tool plate, so nothing in the
#: scene was ever visibly holding the wire. Pinning the *5.0* layout gets a
#: Franka that still has its hand, which is what the drive unit is clamped in.
FRANKA_PANDA_USD = FRANKA_PANDA_HAND_USD

#: The seven arm joints, in the order the USD declares them.
FRANKA_JOINT_NAMES = tuple(f"panda_joint{index}" for index in range(1, 8))

#: The two finger joints. Prismatic, 0 to 40 mm each, opening along the hand's
#: own x. They are driven to a fixed clamp width rather than commanded, since
#: the grip is on a drive unit that is never released mid-procedure.
FRANKA_FINGER_JOINT_NAMES = ("panda_finger_joint1", "panda_finger_joint2")

#: The body the drive unit is clamped to: the hand, between the fingers.
#:
#: This is also the body the wire's reaction is applied to, so it has to be the
#: one physically holding the drive unit rather than a frame near it.
FRANKA_FLANGE_BODY = "panda_hand"

#: Half-width of the fingers' grip, metres. The drive unit's barrel is the
#: thing being gripped, not the 0.89 mm wire, so the fingers close on a barrel
#: rather than to their stop -- fingers touching each other read as an empty
#: hand. Each finger travels this far from closed, so the gap is twice it.
_DRIVE_BARREL_HALF_WIDTH_M = 0.012

#: Home pose: the drive unit already on the introducer, tilted onto the wire.
#:
#: Solved against this arm's own kinematics and joint limits for the twin's
#: femoral access, holding the grip on the access site and the drive axis at
#: :data:`_INTRODUCER_TILT_RAD` out of the vessel. Of the poses that reach it,
#: this is the one furthest from any joint limit, which is what the arm needs
#: to have authority left over for the wire's reaction.
#:
#: Solved for the *whole* orientation, wrist roll included, and not just for
#: the drive axis. Getting the axis right and the roll wrong still parks the
#: hand correctly on paper, but leaves the IK a 2.6 rad turn of
#: ``panda_joint7`` to unwind, and the hand sweeps 13 cm off the access site
#: while it does. The roll here is the one :meth:`_hand_quat_facing` asks for.
#:
#: It is a starting point, not a constraint: the action term resolves the
#: introducer from the rod at runtime and walks the arm onto it, so a twin with
#: a different access site still gets held correctly. Starting near the answer
#: keeps that walk short.
#:
#: The pose it replaced sat at ``panda_joint2 = 1.74`` against a 1.7628 limit.
#: The shoulder was already at its stop on the first frame, so the IK had
#: nowhere to go and parked the hand half a metre above the patient.
FRANKA_HOME_JOINT_POS = {
    "panda_joint1": 0.280,
    "panda_joint2": 1.022,
    "panda_joint3": 0.025,
    "panda_joint4": -1.406,
    "panda_joint5": -0.736,
    "panda_joint6": 1.882,
    "panda_joint7": 0.928,
    # Closed on the drive unit's barrel from the first frame. The hand is never
    # empty in this procedure, so there is no open pose to start from.
    "panda_finger_joint1": _DRIVE_BARREL_HALF_WIDTH_M,
    "panda_finger_joint2": _DRIVE_BARREL_HALF_WIDTH_M,
}

#: Datasheet torque limits: the four proximal joints carry 87 Nm, the three
#: distal ones 12 Nm.
_PROXIMAL_EFFORT_LIMIT_NM = 87.0
_DISTAL_EFFORT_LIMIT_NM = 12.0

#: Datasheet velocity limits, rad/s.
_PROXIMAL_VELOCITY_LIMIT_RAD_S = 2.175
_DISTAL_VELOCITY_LIMIT_RAD_S = 2.610

#: Position gains for an arm that is position-servo'd by differential IK rather
#: than force-controlled. The stock 80/4 gains are tuned for compliant contact
#: work and let the flange lag the commanded pose by enough to lose wire feed.
_SERVO_STIFFNESS = 400.0
_SERVO_DAMPING = 80.0

#: Finger limits from the hand's datasheet: 70 N of grip, 0.2 m/s of travel.
_GRIP_FORCE_LIMIT_N = 70.0
_FINGER_VELOCITY_LIMIT_M_S = 0.2
_FINGER_STIFFNESS = 2000.0
_FINGER_DAMPING = 100.0

#: Distance from the ``panda_hand`` frame to the point between the fingertips,
#: along the hand's approach axis. The servo commands the hand, but the thing
#: that has to land on the introducer is the grip, so every target is offset
#: back down the approach axis by this much.
_HAND_TO_GRIP_M = 0.1034

#: Working length of the introducer sheath: how far the drive unit's grip sits
#: back from the vessel it feeds into. A standard femoral sheath.
#:
#: Without it the grip goes onto the wire's proximal particle itself, and that
#: particle is *inside the patient* -- the vessel access on this twin is 4.5 cm
#: below the skin. The fingers were closing underneath the surface, holding a
#: wire in the middle of the tissue.
#:
#: A sheath is what makes the real procedure work: the tube crosses the skin
#: and the hub stays outside, so the drive never touches the patient. At this
#: length and the tilt below, the grip clears the surface by 3.3 cm and the
#: lowest joint on the arm sits at 1.09 m, well above the 0.91 m skin.
_SHEATH_LENGTH_M = 0.11

#: How long the arm takes to bring the grip from its spawn pose onto the
#: introducer, in seconds. The differential IK is a local linearisation, so
#: handing it the full error at once asks for a joint step it cannot take
#: accurately; the arm swings hard and overshoots. Approaching over a fixed
#: duration keeps every IK step inside the range the linearisation holds over.
_APPROACH_S = 1.0

#: How far the drive unit is tilted up out of the vessel it feeds into.
#:
#: This is the femoral puncture angle, and it is also the only way the arm can
#: hold the site at all. Asking the hand to point *along* the wire puts it
#: inline with the table's long axis while the cart stands beside the table,
#: and that pose is outside the arm's dexterous workspace: a search over the
#: null space finds no solution within the joint limits at any base yaw, and
#: the arm ends up pinned against ``panda_joint2``'s stop half a metre high.
#:
#: Tilting up recovers it with room to spare -- 0.64 rad of margin on the
#: nearest joint limit at this angle, against none at all when parallel. It
#: also lifts the whole arm clear of the patient: every joint origin stays
#: above 1.01 m, where the patient's surface is 0.91 m.
_INTRODUCER_TILT_RAD = math.radians(45.0)


FRANKA_PANDA_CATHETER_CFG = ArticulationCfg(
    spawn=sim_utils.UsdFileCfg(
        # The stock spawner, unlike the healthcare bundle's Panda this
        # replaced. That asset needed a local overlay before Newton would
        # build a model from it -- its wrist joints were authored with parent
        # and child swapped, and nothing grounded ``panda_link0``, so it came
        # up floating and the IK steered with the wrong Jacobian columns. This
        # one ships a world-grounded ``rootJoint`` and no reversed joints, so
        # every pass in that overlay is a no-op here.
        usd_path=FRANKA_PANDA_USD,
        activate_contact_sensors=False,
        # Newton reads the backend-agnostic schemas; the PhysX-prefixed configs
        # in ``embodiments.franka`` describe a solver this scene does not run.
        rigid_props=sim_utils.RigidBodyPropertiesCfg(
            # No ``disable_gravity`` here, despite this arm needing exactly
            # that. It is a PhysX-namespace attribute and this scene runs
            # Newton, which has no per-body gravity to disable -- authoring it
            # reads as a solved problem while the arm goes on sagging. The
            # scene takes gravity out of the world instead; see
            # ``EndoluminalNavigationArmScene.configure_env_cfg``.
            max_depenetration_velocity=5.0,
        ),
        articulation_props=sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=True,
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=0,
        ),
        semantic_tags=[("class", "robot")],
    ),
    init_state=ArticulationCfg.InitialStateCfg(joint_pos=dict(FRANKA_HOME_JOINT_POS)),
    actuators={
        "panda_shoulder": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[1-4]"],
            effort_limit_sim=_PROXIMAL_EFFORT_LIMIT_NM,
            velocity_limit_sim=_PROXIMAL_VELOCITY_LIMIT_RAD_S,
            stiffness=_SERVO_STIFFNESS,
            damping=_SERVO_DAMPING,
        ),
        "panda_forearm": ImplicitActuatorCfg(
            joint_names_expr=["panda_joint[5-7]"],
            effort_limit_sim=_DISTAL_EFFORT_LIMIT_NM,
            velocity_limit_sim=_DISTAL_VELOCITY_LIMIT_RAD_S,
            stiffness=_SERVO_STIFFNESS,
            damping=_SERVO_DAMPING,
        ),
        "panda_hand": ImplicitActuatorCfg(
            joint_names_expr=["panda_finger_joint.*"],
            # Datasheet grip force, shared across the pair.
            effort_limit_sim=_GRIP_FORCE_LIMIT_N,
            velocity_limit_sim=_FINGER_VELOCITY_LIMIT_M_S,
            # Stiff, because the fingers hold a clamp width and never regrasp.
            # Slack fingers drift open under the wire's reaction and the hand
            # reads as dropping the drive unit.
            stiffness=_FINGER_STIFFNESS,
            damping=_FINGER_DAMPING,
        ),
    },
    soft_joint_pos_limit_factor=1.0,
)


def make_franka_panda_catheter_cfg() -> ArticulationCfg:
    """Return a copy of the catheter-carrying Franka Panda articulation config."""
    return FRANKA_PANDA_CATHETER_CFG.copy()


#: Where the cart stands relative to the access site, in the floor plane only.
#: The arm is carried on a wheeled column cart parked on the operator's side of
#: the table, which is -Y: the C-arm pedestal, boom and orbit ring all live at
#: +Y, so an arm placed there would be inside the gantry's sweep.
#:
#: The Y reach is set by the table, not by preference. The slab spans
#: ``y = -0.28 .. 0.28``, so a 0.34 m cart has to park past -0.44 to stand clear
#: of it; -0.616 from an access site near the midline leaves roughly 0.1 m of
#: gap. Pulling the cart further back only costs reach.
ARM_CART_XY_OFFSET_M = (0.0, -0.616)

#: Top of the cart column, and therefore the arm base, in world metres. This is
#: deliberately absolute rather than relative to the access site: a cart stands
#: on the floor, so its height cannot follow the patient up and down. It clears
#: both the table top (0.77) and the patient (0.91) so the arm reaches *down*
#: onto the access site instead of having to climb over the table edge.
ARM_BASE_HEIGHT_M = 0.95

#: Footprint and standing height of the cart column. The top face meets
#: ``ARM_BASE_HEIGHT_M`` and the bottom sits on the procedure floor inset.
ARM_CART_FOOTPRINT_M = (0.34, 0.34)
_PROCEDURE_FLOOR_TOP_M = 0.028

#: Base yaw. The asset's joint-1 zero faces +X; the table is at +Y from the
#: cart, so the whole arm is turned a quarter turn about Z to face its work.
#:
#: The component order is (x, y, z, w), which is what this scene's other
#: ``init_state.rot`` values are written in -- every prop in the cath lab
#: carries ``(0, 0, 0, 1)``, which is the identity only under this order. Read
#: as Isaac Lab's documented (w, x, y, z) it would be a 180-degree flip on the
#: patient table. Getting this backwards is silent rather than fatal: a
#: quarter turn authored as (w, x, y, z) is a quarter turn about *X*, which
#: lays the arm on its side pointing away from the table.
ARM_BASE_YAW_QUAT = (0.0, 0.0, 0.7071067811865476, 0.7071067811865476)

#: Recorded joint names contributed by the arm, in ``FRANKA_JOINT_NAMES`` order.
ARM_STATE_NAMES = tuple(f"arm.{joint}" for joint in FRANKA_JOINT_NAMES)


class ArmDrivenCatheterAction(ActionTerm):
    """Hold the drive unit on the access site and feed the wire through it.

    The arm's contribution is positional. It is asked to park the drive unit on
    the introducer and hold it there, and the wire's entry point and heading are
    carried in the flange frame, so an arm that drifts, sags, or is pushed off
    the site takes the whole wire with it and feels the wire's reaction at a
    defined point. Insertion itself is a roller command and goes to the rod
    directly.

    The rejected alternative was to read insertion off the flange's own travel.
    That ties the wire's reach to the arm's: the rod is a fixed-length 0.4 m
    stick, and this arm has nothing like 0.4 m of straight-line travel from its
    home pose, so most of the rod would have been unreachable.
    """

    cfg: ArmDrivenCatheterActionCfg

    def __init__(self, cfg: ArmDrivenCatheterActionCfg, env: Any):
        super().__init__(cfg, env)
        if not isinstance(self._asset, XpbdCatheterAsset):
            raise TypeError(f"asset {cfg.asset_name!r} must be XpbdCatheterAsset")
        self._robot = env.scene[cfg.robot_name]

        self._joint_ids, self._joint_names = self._robot.find_joints(list(cfg.arm_joint_names))
        body_ids, body_names = self._robot.find_bodies(cfg.flange_body_name)
        if len(body_ids) != 1:
            raise ValueError(f"expected exactly one body matching {cfg.flange_body_name!r}, found {body_names}")
        self._body_idx = body_ids[0]
        # A fixed base contributes no Jacobian row, so the flange's row sits one
        # earlier than its body index; the joint columns shift the other way by
        # however many base DOFs the articulation has.
        self._jacobi_body_idx = self._body_idx - 1 if self._robot.is_fixed_base else self._body_idx
        self._jacobi_joint_ids = [index + self._robot.num_base_dofs for index in self._joint_ids]

        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Integrated here for the same reason as on the plain catheter: the tip's
        # bend is a rest shape the solver holds, not a rate it consumes, so
        # releasing the key has to leave the shape where it was.
        self._tip_bend_angle = torch.zeros(self.num_envs, device=self.device)

        self._ik = DifferentialIKController(cfg=cfg.controller, num_envs=self.num_envs, device=self.device)
        self._introducer = FlangeMountedIntroducer(
            IntroducerDriveSpec(
                max_insertion_velocity_mps=float(cfg.max_insertion_velocity_mps),
                max_rotation_rate_radps=float(cfg.max_rotation_rate_radps),
                travel_limit_m=float(cfg.travel_limit_m),
            ),
            num_envs=self.num_envs,
            device=self.device,
        )
        self._lumen = self._build_lumen_clamp(cfg)

        # The fingers are held by writing a target every step rather than by
        # the spawn pose alone. ``init_state.joint_pos`` seeds the measured
        # angle but leaves the *target* at zero, so a hand posed open closes
        # itself over the first few steps and the drive unit falls out of it.
        self._finger_ids, _ = self._robot.find_joints(list(cfg.finger_joint_names))
        self._grip = torch.full(
            (self.num_envs, len(self._finger_ids)), float(cfg.grip_half_width_m), device=self.device
        )

        # Where the introducer is only becomes knowable once the rod exists, so
        # the pose the arm is asked to hold is resolved on the first step it
        # can be, and the arm is walked onto it from wherever it spawned.
        self._target_pos = torch.zeros((self.num_envs, 3), device=self.device)
        self._target_quat = torch.zeros((self.num_envs, 4), device=self.device)
        self._target_quat[:, 0] = 1.0
        self._target_latched = False
        # What the servo is actually given this step. It equals the target once
        # the arm has arrived, and lags it during the approach.
        self._command_pos = torch.zeros_like(self._target_pos)
        self._command_quat = torch.zeros_like(self._target_quat)
        self._command_quat[:, 0] = 1.0
        self._approach_s = 0.0

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def tip_bend_angle(self) -> torch.Tensor:
        """Current absolute tip bend per env, in radians."""
        return self._tip_bend_angle

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions[:, 0] = torch.clamp(
            actions[:, 0], -float(self.cfg.max_insertion_velocity_mps), float(self.cfg.max_insertion_velocity_mps)
        )
        self._processed_actions[:, 1] = torch.clamp(
            actions[:, 1], -float(self.cfg.max_rotation_rate_radps), float(self.cfg.max_rotation_rate_radps)
        )
        self._processed_actions[:, 2] = torch.clamp(
            actions[:, 2], -float(self.cfg.max_tip_bend_rate_radps), float(self.cfg.max_tip_bend_rate_radps)
        )

    def apply_actions(self) -> None:
        dt = float(self._env.physics_dt)
        flange_pos, flange_quat = self._flange_pose_world()

        frame = self._proximal_frame()
        if frame is None:
            # The rod is not built yet, so there is no introducer to go to.
            # Hold the spawn pose rather than servoing at a stale target.
            if not self._target_latched:
                self._command_pos.copy_(flange_pos)
                self._command_quat.copy_(flange_quat)
            self._servo_arm()
            self._hold_the_grip()
            return
        root_pos, root_quat, tangent = frame

        # Where the grip has to end up is the introducer, which is the wire's
        # entry -- not wherever the home joint pose happened to put the hand.
        # Latching the reached pose is what left the hand holding nothing 30 cm
        # away, since the home pose is an IK solution posed for clearance and
        # was never solved against the access site.
        if not self._target_latched:
            self._aim_at_the_introducer(flange_pos, flange_quat, root_pos, tangent)
        self._advance_the_approach(dt)

        # Once parked, the commanded pose does not move: the introducer is
        # taped to the patient and the wire runs through it. What the servo
        # still buys is resistance, since the reaction coming back up the wire
        # has to be held against.
        self._servo_arm()
        self._hold_the_grip()

        insertion, rotation = self._introducer.advance(self._processed_actions[:, 0], self._processed_actions[:, 1], dt)
        # The flange moves the length of the approach before it parks, and the
        # introducer transports the wire rigidly with it. Reading transport
        # while the arm is still on its way would drag the wire across the
        # patient behind a hand that has not reached the access site yet.
        transport = self._introducer.transport(flange_pos) if self._parked else torch.zeros_like(flange_pos)
        root_target, quat_target = self._introducer.root_target(
            root_pos,
            root_quat,
            tangent,
            transport,
            insertion,
            rotation,
            dt,
        )
        if self._lumen is not None:
            root_target = self._lumen.project(root_target)
        self._asset.place_proximal(root_target, quat_target, torch.stack((insertion, rotation), dim=-1), dt)
        limit = float(self.cfg.max_tip_bend_rad)
        # In place so the buffer handed to the solver keeps its storage.
        self._tip_bend_angle.add_(self._processed_actions[:, 2] * dt).clamp_(-limit, limit)
        self._asset.set_tip_bend(self._tip_bend_angle)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
        self._processed_actions[selected] = 0.0
        # Resolve the introducer again from wherever the rod lands after the
        # reset, and walk onto it again from the spawn pose. The wire retracts;
        # the drive's own geometry is hardware and stays.
        self._target_latched = False
        self._approach_s = 0.0
        # A reset returns a straight wire, so the steer goes with it.
        self._tip_bend_angle[selected] = 0.0
        self._introducer.reset(None if env_ids is None else torch.as_tensor(env_ids, device=self.device))

    def joint_state(self) -> JointState:
        """Report the arm's servo'd joints as part of the procedure state."""
        pos = self._robot.data.joint_pos.torch[:, self._joint_ids]
        vel = self._robot.data.joint_vel.torch[:, self._joint_ids]
        return JointState(
            pos=pos.detach().cpu().numpy().astype(np.float32, copy=False),
            vel=vel.detach().cpu().numpy().astype(np.float32, copy=False),
            names=tuple(f"arm.{name}" for name in self._joint_names),
        )

    def _build_lumen_clamp(self, cfg: ArmDrivenCatheterActionCfg) -> LumenClamp | None:
        """The vessel the prescribed proximal particle has to stay inside.

        Absent without a patient twin, since there is then no lumen to speak of
        and a made-up one would constrain the wire to a vessel that does not
        exist.
        """
        path = cfg.lumen_path_world_m
        radii = cfg.lumen_radii_m
        if not path or not radii:
            return None
        return LumenClamp(
            torch.tensor(path, dtype=torch.float32),
            torch.tensor(radii, dtype=torch.float32),
            margin_m=float(cfg.lumen_margin_m),
            device=self.device,
        )

    def _proximal_frame(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None:
        """The rod's proximal pose and first-segment direction, if the rod is up.

        The rod solver is built by the Newton manager, which may not have run
        when the first action is applied. Reporting nothing is the honest answer
        for that step; guessing a pose would put the wire somewhere arbitrary.
        """
        try:
            return self._asset.proximal_frame()
        except RuntimeError:
            return None

    def _flange_pose_world(self) -> tuple[torch.Tensor, torch.Tensor]:
        data = self._robot.data
        return data.body_pos_w.torch[:, self._body_idx], data.body_quat_w.torch[:, self._body_idx]

    @property
    def _parked(self) -> bool:
        """Has the grip finished travelling onto the introducer?"""
        return self._target_latched and self._approach_s >= _APPROACH_S

    def _aim_at_the_introducer(
        self,
        flange_pos: torch.Tensor,
        flange_quat: torch.Tensor,
        root_pos: torch.Tensor,
        tangent: torch.Tensor,
    ) -> None:
        """Resolve the pose that puts the grip on the wire's entry, and set off.

        The sheath's bore runs to the wire's proximal end, so the heading comes
        from the rod, while the grip goes on the sheath's hub -- a sheath's
        length back up that heading, outside the patient. Both are read from
        the rod rather than configured, which is what keeps this correct when
        the patient twin moves the access site.
        """
        wire = torch.nn.functional.normalize(tangent, dim=-1, eps=1e-9)
        # A rod whose first segment is degenerate reports no heading. Keeping
        # the hand's current axis is the only answer that is not invented.
        wire = torch.where(wire.norm(dim=-1, keepdim=True) > 0.5, wire, self._approach_axis(flange_quat))
        approach = self._tilted_out_of(wire)

        self._target_pos.copy_(root_pos - approach * (_SHEATH_LENGTH_M + _HAND_TO_GRIP_M))
        self._target_quat.copy_(self._hand_quat_facing(approach))
        self._command_pos.copy_(flange_pos)
        self._command_quat.copy_(flange_quat)
        self._approach_s = 0.0
        self._target_latched = True

    def _tilted_out_of(self, wire: torch.Tensor) -> torch.Tensor:
        """Raise the drive unit's axis off the wire by the puncture angle.

        The tilt is taken in the vertical plane through the wire, so the drive
        comes down onto the site from above and along the vessel rather than
        across it. Its bore still meets the wire at the entry: only the hand
        is angled, and the grip lands on the access site either way.
        """
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(wire)
        tilted = wire * math.cos(_INTRODUCER_TILT_RAD) - up * math.sin(_INTRODUCER_TILT_RAD)
        return torch.nn.functional.normalize(tilted, dim=-1, eps=1e-9)

    def _approach_axis(self, hand_quat: torch.Tensor) -> torch.Tensor:
        """The hand's own ``+z``, which is the direction it reaches along."""
        local = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand(hand_quat.shape[0], 3)
        return math_utils.quat_apply(hand_quat, local)

    def _hand_quat_facing(self, feed: torch.Tensor) -> torch.Tensor:
        """A hand orientation whose approach axis lies along ``feed``.

        One degree of freedom is left over, since spinning the hand about its
        own approach axis still points it down the wire. It is spent laying the
        fingers' travel axis horizontal, so the hand closes across the wire the
        way an operator's fingers would rather than over and under it.
        """
        up = torch.tensor([0.0, 0.0, 1.0], device=self.device).expand_as(feed)
        across = torch.linalg.cross(up, feed)
        # Degenerate only for a vertical wire, where "horizontal" picks out no
        # particular direction and any perpendicular will do.
        fallback = torch.linalg.cross(torch.tensor([1.0, 0.0, 0.0], device=self.device).expand_as(feed), feed)
        across = torch.where(across.norm(dim=-1, keepdim=True) > 1e-6, across, fallback)
        across = torch.nn.functional.normalize(across, dim=-1, eps=1e-9)

        rotation = torch.stack((across, torch.linalg.cross(feed, across), feed), dim=-1)
        return math_utils.quat_from_matrix(rotation)

    def _advance_the_approach(self, dt: float) -> None:
        """Walk the commanded pose from the spawn pose onto the introducer."""
        if self._parked:
            return
        self._approach_s = min(self._approach_s + dt, _APPROACH_S)
        # Smoothstep rather than linear, so the arm leaves and arrives at rest
        # instead of stepping straight to full commanded speed.
        alpha = self._approach_s / _APPROACH_S
        alpha = alpha * alpha * (3.0 - 2.0 * alpha)

        self._command_pos.lerp_(self._target_pos, alpha)
        self._command_quat.copy_(self._slerp(self._command_quat, self._target_quat, alpha))

    @staticmethod
    def _slerp(start: torch.Tensor, end: torch.Tensor, alpha: float) -> torch.Tensor:
        """Normalized linear blend between two w-first quaternions.

        The two poses are close together at every step of the approach, where
        a normalized blend and a true slerp are indistinguishable. Flipping the
        far end onto the near hemisphere first is the part that matters: without
        it the blend takes the long way round and the wrist spins a full turn.
        """
        end = torch.where((start * end).sum(dim=-1, keepdim=True) < 0.0, -end, end)
        return torch.nn.functional.normalize(torch.lerp(start, end, alpha), dim=-1)

    def _hold_the_grip(self) -> None:
        """Command the fingers onto the drive unit's barrel."""
        if self._finger_ids:
            self._robot.set_joint_position_target_index(target=self._grip, joint_ids=self._finger_ids)

    def _servo_arm(self) -> None:
        """Drive one differential-IK step toward the commanded flange pose."""
        data = self._robot.data
        root_pos = data.root_pos_w.torch
        root_quat = data.root_quat_w.torch

        target_pos_b, target_quat_b = math_utils.subtract_frame_transforms(
            root_pos, root_quat, self._command_pos, self._command_quat
        )
        self._ik.set_command(torch.cat((target_pos_b, target_quat_b), dim=-1))

        flange_pos, flange_quat = self._flange_pose_world()
        flange_pos_b, flange_quat_b = math_utils.subtract_frame_transforms(root_pos, root_quat, flange_pos, flange_quat)
        if float(flange_quat_b.norm()) == 0.0:
            return

        joint_pos = data.joint_pos.torch[:, self._joint_ids]
        joint_pos_des = self._ik.compute(flange_pos_b, flange_quat_b, self._jacobian_root_frame(), joint_pos)
        self._robot.set_joint_position_target_index(target=joint_pos_des, joint_ids=self._joint_ids)

    def _jacobian_root_frame(self) -> torch.Tensor:
        data = self._robot.data
        jacobian = data.body_link_jacobian_w.torch[:, self._jacobi_body_idx, :, self._jacobi_joint_ids]
        base_rot = math_utils.matrix_from_quat(math_utils.quat_inv(data.root_quat_w.torch))
        jacobian[:, :3, :] = torch.bmm(base_rot, jacobian[:, :3, :])
        jacobian[:, 3:, :] = torch.bmm(base_rot, jacobian[:, 3:, :])
        return jacobian


@configclass
class ArmDrivenCatheterActionCfg(ActionTermCfg):
    class_type: type[ArmDrivenCatheterAction] = ArmDrivenCatheterAction
    asset_name: str = "catheter"
    robot_name: str = "robot"
    flange_body_name: str = FRANKA_FLANGE_BODY
    arm_joint_names: tuple[str, ...] = FRANKA_JOINT_NAMES
    #: Driven to a constant width, not commanded: the hand holds one drive unit
    #: for the whole procedure and there is nothing to regrasp.
    finger_joint_names: tuple[str, ...] = FRANKA_FINGER_JOINT_NAMES
    grip_half_width_m: float = _DRIVE_BARREL_HALF_WIDTH_M
    #: How far the rollers may feed. The rod is a fixed-length stick whose root
    #: is what moves, so feeding past its own length would drag the whole rod
    #: through the patient; this leaves a segment or two proximal of the site.
    travel_limit_m: float = 0.36
    #: The vessel the proximal particle has to stay inside, and how wide it is
    #: at each sample. Both come from the patient centerline and stay empty
    #: without a twin, which leaves the clamp off rather than inventing a lumen.
    lumen_path_world_m: tuple[tuple[float, float, float], ...] = ()
    lumen_radii_m: tuple[float, ...] = ()
    #: Kept clear of the vessel wall. The catheter's own radius, so what touches
    #: the wall is the wire's surface rather than its axis.
    lumen_margin_m: float = 0.0005
    #: Ceiling on commanded insertion, and what the fluoroscopy velocity slider
    #: is scaled against. Bounded by contact rather than by anatomy: physics
    #: runs at 240 Hz, so this spends 0.25 mm per physics step against a 0.5 mm
    #: particle radius. Half a radius per step still leaves the vessel contact
    #: something to catch; raising it much further lets the wire step past a
    #: wall between solves.
    max_insertion_velocity_mps: float = 0.060
    max_rotation_rate_radps: float = 1.5
    #: Held equal to the plain catheter's ceilings: the same keyboard maps both
    #: embodiments, so a mismatch would silently clip one of them. See
    #: ``CatheterVelocityActionCfg`` for what bounds the values.
    max_tip_bend_rate_radps: float = 1.5
    max_tip_bend_rad: float = 1.5
    controller: DifferentialIKControllerCfg = DifferentialIKControllerCfg(
        command_type="pose",
        use_relative_mode=False,
        ik_method="dls",
        ik_params={"lambda_val": 0.20},
    )


class ArmCatheterCArmJointStateProvider:
    """Catheter, C-arm, and arm joints as one recorded procedure state.

    Order matters and is fixed: the catheter and C-arm columns come first, so a
    recording made with the arm keeps the plain catheter's layout in its leading
    columns and only appends the arm. ``franka_catheter.yaml`` declares the same
    order, and the LeRobot converter silently falls back to positional names if
    the widths ever disagree.
    """

    def __init__(
        self,
        catheter: XpbdCatheterAsset,
        carm_orbit: CArmOrbitAction,
        arm: ArmDrivenCatheterAction,
    ) -> None:
        self._catheter = catheter
        self._carm_orbit = carm_orbit
        self._arm = arm

    def joint_state(self) -> JointState:
        catheter = self._catheter.joint_state()
        carm = self._carm_orbit.joint_state()
        arm = self._arm.joint_state()
        return JointState(
            pos=np.concatenate((catheter.pos, carm.pos, arm.pos), axis=-1),
            vel=np.concatenate((catheter.vel, carm.vel, arm.vel), axis=-1),
            names=(*catheter.names, *carm.names, *arm.names),
        )


@configclass
class _FrankaCatheterSceneCfg(_CatheterSceneCfg):
    """Catheter scene plus the cart and the arm that carries the drive."""

    robot = make_franka_panda_catheter_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")

    #: The column the arm is bolted to. Visual only: it carries no collider,
    #: because the arm is position-servo'd and never leans on its own cart, and
    #: a static box in the Newton model would only add contact pairs. Its job is
    #: to make the scene legible -- an arm floating at 0.95 m with nothing under
    #: it reads as a placement bug even when the placement is right.
    arm_cart = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/ArmCart",
        spawn=sim_utils.CuboidCfg(
            size=(*ARM_CART_FOOTPRINT_M, ARM_BASE_HEIGHT_M - _PROCEDURE_FLOOR_TOP_M),
            visual_material=sim_utils.PreviewSurfaceCfg(
                diffuse_color=(0.30, 0.33, 0.36), roughness=0.45, metallic=0.35
            ),
        ),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(0.0, 0.0, 0.0)),
    )


@configclass
class _FrankaCatheterActionsCfg:
    catheter: ArmDrivenCatheterActionCfg = ArmDrivenCatheterActionCfg()
    carm_orbit: CArmOrbitActionCfg = CArmOrbitActionCfg()


class FrankaCatheterEmbodiment(CatheterEmbodiment):
    """Catheter driven through a MJWarp-simulated Franka Panda flange."""

    name = "franka_catheter"
    tags: ClassVar[list[str]] = ["embodiment", "medical", "catheter", "franka"]

    def __init__(
        self,
        patient_twin_manifest: str | None = None,
        cart_xy_offset_m: tuple[float, float] = ARM_CART_XY_OFFSET_M,
        base_height_m: float = ARM_BASE_HEIGHT_M,
    ) -> None:
        super().__init__(patient_twin_manifest=patient_twin_manifest)
        # The parent resolved the access site and the C-arm isocenter from the
        # patient centerline. Both configs are being replaced, so carry those
        # two facts across rather than recomputing or losing them.
        access_site = self.scene_config.catheter_root.init_state.pos
        isocenter = self.action_config.carm_orbit.isocenter_world_m

        self.scene_config = _FrankaCatheterSceneCfg()
        self.scene_config.catheter_root.init_state.pos = access_site
        self.scene_config.robot = make_franka_panda_catheter_cfg().replace(prim_path="{ENV_REGEX_NS}/Robot")

        self.action_config = _FrankaCatheterActionsCfg()
        self.action_config.carm_orbit.isocenter_world_m = isocenter
        # No insertion axis to hand over: the drive feeds along the wire's own
        # first segment, which the centerline already seeded along the vessel.
        # A configured axis would be a second opinion about the vessel's
        # direction that goes stale the moment the wire rounds a bend.
        #
        # The lumen is handed over, though. The proximal particle is prescribed,
        # so wall contact never acts on it and it is the one part of the rod that
        # can leave the vessel unchallenged.
        if self.rod_spec.initial_path_world_m and self.rod_spec.lumen_radii_m:
            self.action_config.catheter.lumen_path_world_m = self.rod_spec.initial_path_world_m
            self.action_config.catheter.lumen_radii_m = self.rod_spec.lumen_radii_m
            self.action_config.catheter.lumen_margin_m = float(self.rod_spec.radius_m)

        self._cart_xy_offset_m = tuple(float(value) for value in cart_xy_offset_m)
        self._base_height_m = float(base_height_m)
        self._park_the_cart_beside_the_table()

        # An articulation in the model leaves the rod-only solver without a
        # rigid integrator, which is what selects the coupled MJWarp + XPBD one.
        self.rod_spec.rigid_bodies_enabled = True
        # The drive unit is bolted to the flange, so that is the body the wire
        # hangs off and pushes back on. Naming it turns the contact two-way,
        # which is what lets the arm feel insertion resistance rather than
        # servoing as if the catheter weighed nothing.
        self.rod_spec.drive_body_name = FRANKA_FLANGE_BODY
        self.rod_spec.drive_mount_local = (0.0, 0.0, _HAND_TO_GRIP_M)
        self.rod_spec.__post_init__()

    @property
    def arm_base_world_m(self) -> tuple[float, float, float]:
        return tuple(float(value) for value in self.scene_config.robot.init_state.pos)

    def _park_the_cart_beside_the_table(self) -> None:
        """Stand the cart on the floor beside the access site and mount the arm.

        Only X and Y follow the access site. The height does not: the cart rests
        on the floor, so moving to a different patient rolls the cart along the
        table rather than levitating it. Deriving the base height from the
        access site is exactly the bug this replaces -- it left the arm hanging
        in mid-air with its column nowhere and the flange inside the table.
        """
        access_site = np.asarray(self.scene_config.catheter_root.init_state.pos, dtype=np.float64)
        base_x = access_site[0] + self._cart_xy_offset_m[0]
        base_y = access_site[1] + self._cart_xy_offset_m[1]

        self.scene_config.robot.init_state.pos = (base_x, base_y, self._base_height_m)
        self.scene_config.robot.init_state.rot = ARM_BASE_YAW_QUAT

        column_height = self._base_height_m - _PROCEDURE_FLOOR_TOP_M
        self.scene_config.arm_cart.spawn.size = (*ARM_CART_FOOTPRINT_M, column_height)
        self.scene_config.arm_cart.init_state.pos = (
            base_x,
            base_y,
            _PROCEDURE_FLOOR_TOP_M + column_height / 2.0,
        )

    def modify_env_cfg(self, env_cfg: Any) -> Any:
        """Keep the parent's rates, rebuild physics, and seed the arm's pose."""
        env_cfg = super().modify_env_cfg(env_cfg)
        env_cfg.sim.physics = newton_physics_cfg(self.rod_spec)
        self._seed_the_arm_pose_on_reset(env_cfg)
        return env_cfg

    @staticmethod
    def _seed_the_arm_pose_on_reset(env_cfg: Any) -> None:
        """Write the arm's configured home pose into the simulation on reset.

        ``ArticulationCfg.init_state.joint_pos`` only populates
        ``default_joint_pos``; something still has to push those defaults into
        the articulation, and this scene's event config is otherwise empty. The
        armless catheter scene never needed one because it has no articulation,
        so the omission is invisible until an arm arrives -- and then it is
        invisible again in the numbers, because ``default_joint_pos`` reads back
        correct while the simulation holds all-zeros. The symptom is a Franka
        standing bolt upright through the patient table.

        Zero-width ranges make this a plain "reset to default" rather than a
        randomisation; the arm is servo'd to the introducer, so a jittered start
        would only add settling time before the first useful frame.
        """
        from isaaclab.envs import mdp
        from isaaclab.managers import EventTermCfg, SceneEntityCfg

        env_cfg.events.reset_arm_to_home = EventTermCfg(
            func=mdp.reset_joints_by_offset,
            mode="reset",
            params={
                "position_range": (0.0, 0.0),
                "velocity_range": (0.0, 0.0),
                "asset_cfg": SceneEntityCfg("robot"),
            },
        )


__all__ = [
    "ARM_BASE_HEIGHT_M",
    "ARM_BASE_YAW_QUAT",
    "ARM_CART_FOOTPRINT_M",
    "ARM_CART_XY_OFFSET_M",
    "ARM_STATE_NAMES",
    "FRANKA_FLANGE_BODY",
    "FRANKA_JOINT_NAMES",
    "FRANKA_PANDA_CATHETER_CFG",
    "ArmCatheterCArmJointStateProvider",
    "ArmDrivenCatheterAction",
    "ArmDrivenCatheterActionCfg",
    "FrankaCatheterEmbodiment",
    "make_franka_panda_catheter_cfg",
]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-axis catheter embodiment backed by the i4h Warp XPBD solver."""

from __future__ import annotations

import json
import math
from collections.abc import Sequence
from typing import Any, ClassVar

import isaaclab.sim as sim_utils
import numpy as np
import torch
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass

from i4h_arena.medical.centerline import ordered_centerline_lumen
from i4h_arena.medical.newton_catheter_physics import (
    DEFAULT_NUM_SEGMENTS,
    CatheterRodHandle,
    CatheterRodSpec,
    cleanup_sweeps_override,
    containment_stage_override,
    newton_physics_cfg,
    rod_damping_override,
    segment_count_override,
)
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.medical.xpbd_catheter import XpbdCatheterAsset, XpbdCatheterAssetCfg
from i4h_common.types import JointState


def reference_initial_catheter_length_m(twin: PatientTwin, *, fallback_m: float) -> float:
    """Match the reference viewport's 15%-to-80% CT-width initialization."""
    metadata_path = twin.artifacts.get("volume_metadata")
    if metadata_path is None:
        return float(fallback_m)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    shape_zyx = np.asarray(metadata.get("shape_zyx"), dtype=np.float64)
    spacing_zyx_mm = np.asarray(metadata.get("spacing_zyx_mm"), dtype=np.float64)
    if shape_zyx.shape != (3,) or spacing_zyx_mm.shape != (3,):
        raise ValueError("volume metadata must contain three-value shape_zyx and spacing_zyx_mm")
    length_m = 0.65 * float(shape_zyx[2] * spacing_zyx_mm[2]) * 0.001
    if not np.isfinite(length_m) or length_m <= 0.0:
        raise ValueError("volume metadata produces an invalid catheter initialization length")
    return min(float(fallback_m), length_m)


class CatheterVelocityAction(ActionTerm):
    """Proximal insertion velocity, axial rotation rate, and tip bend rate in SI units.

    The first two terms are rates the solver consumes. The third is different in
    kind: the tip's bend is a rest shape the solver holds, so this term
    integrates the commanded rate into an angle it owns and hands over the angle
    rather than the rate. Steering has to persist between commands the way a
    shaped wire does, and releasing the key has to hold the shape rather than
    let it spring back.

    The bend is about the tip's local X axis. Aiming it at a branch is the
    rotation term's job, which is how a pre-shaped wire is aimed clinically.
    """

    cfg: CatheterVelocityActionCfg

    def __init__(self, cfg: CatheterVelocityActionCfg, env: Any):
        super().__init__(cfg, env)
        if not isinstance(self._asset, XpbdCatheterAsset):
            raise TypeError(f"asset {cfg.asset_name!r} must be XpbdCatheterAsset")
        self._raw_actions = torch.zeros((self.num_envs, 3), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        # Held here rather than on the asset because it is this term's
        # integration of the command, and a reset has to clear it per env.
        self._tip_bend_angle = torch.zeros(self.num_envs, device=self.device)

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
        # Only the two rate terms are the asset's velocity contract; the bend is
        # a separate shape command, so it does not travel through ``advance``.
        self._asset.advance(self._processed_actions[:, :2], dt)
        limit = float(self.cfg.max_tip_bend_rad)
        # In place so the buffer the solver was handed keeps its storage.
        self._tip_bend_angle.add_(self._processed_actions[:, 2] * dt).clamp_(-limit, limit)
        self._asset.set_tip_bend(self._tip_bend_angle)

    def reset(self, env_ids: Sequence[int] | None = None) -> None:
        if env_ids is None:
            self._raw_actions.zero_()
            self._processed_actions.zero_()
            self._tip_bend_angle.zero_()
        else:
            self._raw_actions[env_ids] = 0.0
            self._processed_actions[env_ids] = 0.0
            # A reset returns a straight wire, so the steer has to go with it.
            self._tip_bend_angle[env_ids] = 0.0


@configclass
class CatheterVelocityActionCfg(ActionTermCfg):
    class_type: type[CatheterVelocityAction] = CatheterVelocityAction
    asset_name: str = "catheter"
    #: Kept equal to the arm-driven term's ceiling, since the same fluoroscopy
    #: velocity slider drives both and a mismatch would silently clip one of
    #: them. See that term for what bounds the value.
    max_insertion_velocity_mps: float = 0.060
    max_rotation_rate_radps: float = 1.5
    #: Rate the tip's bend can be steered at. At this ceiling a full deflection
    #: takes about a second, which is deliberate: the tip is shaped against the
    #: bend constraint's rest state, and stepping that faster than the solve
    #: relaxes asks the wire to snap rather than curl.
    max_tip_bend_rate_radps: float = 1.5
    #: Total bend across the tip edges, so roughly a 90-degree hook at the
    #: ceiling. Matches the clamp the reference implementation settled on.
    max_tip_bend_rad: float = 1.5


class CArmOrbitAction(ActionTerm):
    """Rotate the complete source-detector assembly about the patient long axis."""

    cfg: CArmOrbitActionCfg

    def __init__(self, cfg: CArmOrbitActionCfg, env: Any):
        super().__init__(cfg, env)
        self._raw_actions = torch.zeros((self.num_envs, 1), device=self.device)
        self._processed_actions = torch.zeros_like(self._raw_actions)
        self._angle_rad = torch.full(
            (self.num_envs,), float(cfg.initial_orbit_angle_rad), device=self.device, dtype=torch.float32
        )
        self._root_position = torch.tensor(cfg.isocenter_world_m, device=self.device, dtype=torch.float32).repeat(
            self.num_envs, 1
        )
        self._root_orientation = torch.zeros((self.num_envs, 4), device=self.device, dtype=torch.float32)
        self._write_pose()

    @property
    def action_dim(self) -> int:
        return 1

    @property
    def raw_actions(self) -> torch.Tensor:
        return self._raw_actions

    @property
    def processed_actions(self) -> torch.Tensor:
        return self._processed_actions

    @property
    def angle_rad(self) -> torch.Tensor:
        return self._angle_rad

    def process_actions(self, actions: torch.Tensor) -> None:
        self._raw_actions.copy_(actions)
        self._processed_actions.copy_(
            torch.clamp(actions, -float(self.cfg.max_orbit_rate_radps), float(self.cfg.max_orbit_rate_radps))
        )

    def apply_actions(self) -> None:
        self._angle_rad.add_(self._processed_actions[:, 0] * float(self._env.physics_dt))
        self._angle_rad.clamp_(float(self.cfg.min_orbit_angle_rad), float(self.cfg.max_orbit_angle_rad))
        self._write_pose()

    def reset(self, env_ids: Sequence[int] | slice | None = None) -> None:
        selected = slice(None) if env_ids is None else env_ids
        self._raw_actions[selected] = 0.0
        self._processed_actions[selected] = 0.0
        self._angle_rad[selected] = float(self.cfg.initial_orbit_angle_rad)
        self._write_pose()

    def joint_state(self) -> JointState:
        return JointState(
            pos=self._angle_rad[:, None].detach().cpu().numpy().astype(np.float32, copy=False),
            vel=self._processed_actions.detach().cpu().numpy().astype(np.float32, copy=False),
            names=("carm_orbit_rad",),
        )

    def set_orbit_angle(self, angle_rad: float) -> float:
        """Set a named projection angle immediately and return the clamped value."""
        selected = float(np.clip(angle_rad, self.cfg.min_orbit_angle_rad, self.cfg.max_orbit_angle_rad))
        self._angle_rad.fill_(selected)
        self._processed_actions.zero_()
        self._write_pose()
        return selected

    def _write_pose(self) -> None:
        half_angle = 0.5 * self._angle_rad
        self._root_orientation[:, 0] = torch.sin(half_angle)
        self._root_orientation[:, 1:3] = 0.0
        self._root_orientation[:, 3] = torch.cos(half_angle)
        self._asset.set_local_poses(
            translations=self._root_position,
            orientations=self._root_orientation,
        )


@configclass
class CArmOrbitActionCfg(ActionTermCfg):
    class_type: type[CArmOrbitAction] = CArmOrbitAction
    asset_name: str = "carm_orbit_root"
    isocenter_world_m: tuple[float, float, float] = (0.0, 0.0, 0.85)
    initial_orbit_angle_rad: float = math.pi / 4.0
    max_orbit_rate_radps: float = 0.6
    min_orbit_angle_rad: float = -math.pi / 6.0
    max_orbit_angle_rad: float = math.pi / 2.0


class CatheterCArmJointStateProvider:
    """Record catheter virtual joints and C-arm angle as one procedure state."""

    def __init__(self, catheter: XpbdCatheterAsset, carm_orbit: CArmOrbitAction) -> None:
        self._catheter = catheter
        self._carm_orbit = carm_orbit

    def joint_state(self) -> JointState:
        catheter = self._catheter.joint_state()
        carm = self._carm_orbit.joint_state()
        return JointState(
            pos=np.concatenate((catheter.pos, carm.pos), axis=-1),
            vel=np.concatenate((catheter.vel, carm.vel), axis=-1),
            names=(*catheter.names, *carm.names),
        )


@configclass
class _CatheterSceneCfg:
    catheter_root = AssetBaseCfg(
        prim_path="{ENV_REGEX_NS}/Catheter",
        spawn=sim_utils.SphereCfg(radius=0.0001, visible=False),
        init_state=AssetBaseCfg.InitialStateCfg(pos=(-0.11, 0.04, 0.68)),
    )
    catheter = XpbdCatheterAssetCfg(
        prim_path="{ENV_REGEX_NS}/Catheter",
        update_period=0.0,
        debug_vis=True,
    )


@configclass
class _ActionsCfg:
    catheter = CatheterVelocityActionCfg()
    carm_orbit = CArmOrbitActionCfg()


class CatheterEmbodiment:
    """Minimal Arena embodiment whose scene entity is an external rod solver."""

    name = "catheter"
    tags: ClassVar[list[str]] = ["embodiment", "medical", "catheter"]

    def __init__(self, patient_twin_manifest: str | None = None) -> None:
        self.scene_config = _CatheterSceneCfg()
        self.action_config = _ActionsCfg()
        # Rod geometry belongs to the physics spec now that Newton's manager
        # owns the solver; the scene entity only needs the radius it draws.
        self.rod_spec = CatheterRodSpec(
            origin_world_m=(-0.11, 0.04, 0.68),
            radius_m=float(self.scene_config.catheter.radius_m),
            patient_twin_manifest=patient_twin_manifest,
        )
        #: Distal centerline point the tip has to reach, in Isaac world metres.
        #: Stays ``None`` without a twin, where there is no vessel to navigate.
        self.navigation_target_world_m: tuple[float, float, float] | None = None
        self._rod_handle: CatheterRodHandle | None = None
        if patient_twin_manifest is not None:
            self._align_to_patient_centerline(PatientTwin.load(patient_twin_manifest))

    def _align_to_patient_centerline(self, twin: PatientTwin) -> None:
        patient = PatientVolume.load(twin)
        isocenter = patient.volume_mm_to_world(patient.center_xyz_mm)
        self.action_config.carm_orbit.isocenter_world_m = tuple(float(value) for value in isocenter)
        centerline_path = twin.artifacts.get("centerline_points")
        if centerline_path is None:
            return
        points_patient_mm = np.load(centerline_path)
        edges_path = twin.artifacts.get("centerline_edges")
        if edges_path is None:
            return
        edges = np.load(edges_path)
        radii_path = twin.artifacts.get("centerline_radii")
        radii = np.load(radii_path) if radii_path is not None else None
        path_patient_mm, lumen_radii_mm = ordered_centerline_lumen(
            points_patient_mm,
            edges,
            target_spacing_mm=7.5,
            radii_mm=radii,
        )
        path_world_m = twin.patient_mm_to_world(path_patient_mm)
        path_segments = np.linalg.norm(np.diff(path_world_m, axis=0), axis=1)
        path_length = float(np.sum(path_segments))
        length = reference_initial_catheter_length_m(twin, fallback_m=path_length)
        start = path_world_m[0]
        direction = path_world_m[1] - path_world_m[0]
        direction /= np.linalg.norm(direction)

        origin = tuple(float(value) for value in start)
        track_direction = tuple(float(value) for value in direction)
        self.scene_config.catheter_root.init_state.pos = origin
        self.rod_spec.origin_world_m = origin
        self.rod_spec.track_direction_world = track_direction
        self.rod_spec.length_m = length
        # 40 segments over this route put a joint every ~16 mm, too coarse to
        # sit smoothly against the lumen: a joint held one containment state
        # until the solve tipped it into another, which read as the tip jumping
        # about a millimetre inside a single 0.77 s sample while the operator
        # was holding still. Refining removed those jumps outright. The count
        # stays overridable because the step cost is real and the bend
        # stiffness has to be compensated alongside it.
        self.rod_spec.num_segments = segment_count_override() or DEFAULT_NUM_SEGMENTS
        # The centerline seeds the rod's initial shape so the catheter starts
        # inside the lumen. It is no longer resampled every step; containment
        # against the deformable wall is what keeps it there.
        #
        # Containment runs "post", after the constraint solve, because that is
        # the only side the projection survives on, and the cleanup sweeps repair
        # the chords it costs. Measured on the s0011 iliac route: +2.7 to +3.4 mm
        # worst penetration, 2-3 of 41 particles outside, chords 100-112%.
        #
        # "pre" is the better shape and still not usable. It delivers exactly
        # what it promises -- chords land at 100-100% of rest -- but 39 of 41
        # particles end up as much as 74 mm outside the lumen. Writing the rod's
        # real rotational inertia over the solver's identity default does not
        # change that by itself: the two stagings measure 74 mm and 3 mm with the
        # inertia fix in place, the same as without it.
        #
        # Nor does damping rescue it. Running fully quasi-static at damping 1.0,
        # which zeroes velocity and gravity every substep and makes each step a
        # pure geometric projection, measures +74.6 mm and 39 of 41 outside --
        # identical to damping 0.01, across 720 steps of a steady equilibrium.
        # A sparse attraction toward the centerline, re-applied every step, was
        # measured too and came back marginally worse at +74.9 mm and 40 of 41.
        #
        # The reason none of it moves: nothing in the pipeline asks the rod to be
        # curved. The rest shape is straight, and containment is one-sided --
        # acting on a particle only once it is already outside the wall, doing
        # nothing for one inside. So the inward shove and the straightening solve
        # balance tens of millimetres out, and the wire renders as a straight
        # line down the spine. A pre-solve nudge cannot survive a direct solve
        # that lands exactly on the straight manifold, so it never accumulates.
        #
        # That leaves "post" as the configuration that follows the vessel, and
        # closing the gap properly as solver-side work: the solve has to accept
        # a curved rest configuration. The stage switch stays for measurement.
        stage = containment_stage_override() or "post"
        self.rod_spec.containment_stage = stage
        damping = rod_damping_override()
        if damping is not None:
            self.rod_spec.solver_overrides = {
                **self.rod_spec.solver_overrides,
                "linear_damping": damping,
                "angular_damping": damping,
            }
        # The cleanup sweeps exist only to repair the chords that a post-solve
        # projection mangles. Under "pre" there is nothing to repair, and running
        # them would be a third opinion on position competing with the solve.
        sweeps = cleanup_sweeps_override()
        if sweeps is None:
            sweeps = self.rod_spec.containment_cleanup_iterations
        self.rod_spec.containment_cleanup_iterations = sweeps if stage == "post" else 0
        self.rod_spec.initial_path_world_m = tuple(tuple(float(value) for value in point) for point in path_world_m)
        # How wide the vessel is at each of those samples. Carried alongside the
        # path because "on the centerline" is not a testable claim without a
        # tolerance: a prescribed particle exempt from wall contact can sit
        # centimetres outside the lumen and nothing in the path alone objects.
        if lumen_radii_mm is not None:
            self.rod_spec.lumen_radii_m = tuple(float(value) / 1000.0 for value in lumen_radii_mm)
        self.rod_spec.__post_init__()
        # The rod is seeded over the first ``length_m`` of the path, so its tip
        # starts short of the far end and insertion has to cover the remainder.
        self.navigation_target_world_m = tuple(float(value) for value in path_world_m[-1])

    def get_scene_cfg(self) -> Any:
        return self.scene_config

    def get_action_cfg(self) -> Any:
        return self.action_config

    def get_observation_cfg(self) -> None:
        return None

    def get_events_cfg(self) -> Any:
        if self.navigation_target_world_m is None:
            return None
        from i4h_arena.envcfg.endoluminal_navigation import CatheterNavigationEventsCfg

        return CatheterNavigationEventsCfg()

    def get_rewards_cfg(self) -> None:
        return None

    def get_curriculum_cfg(self) -> None:
        return None

    def get_commands_cfg(self) -> None:
        return None

    def get_xr_cfg(self) -> None:
        return None

    def get_recorder_term_cfg(self) -> None:
        return None

    def get_termination_cfg(self) -> Any:
        """Report arrival at the distal centerline as IsaacLab's ``success`` term.

        Omitted without a twin rather than reported as permanently unsatisfied:
        there is no vessel to navigate, so a workflow reading the term gets the
        absent-term zeros instead of a goal it can never meet.
        """
        if self.navigation_target_world_m is None:
            return None
        from i4h_arena.envcfg.endoluminal_navigation import navigation_terminations_cfg

        return navigation_terminations_cfg(self.navigation_target_world_m)

    def modify_env_cfg(self, env_cfg: Any) -> Any:
        env_cfg.sim.dt = 1.0 / 120.0
        # The reference catheter viewport advances controls at 30 Hz. Keep the
        # same control period while retaining 120 Hz XPBD substeps.
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        env_cfg.scene.replicate_physics = False

        self.rod_spec.num_envs = int(getattr(env_cfg.scene, "num_envs", 1))
        self.rod_spec.device = str(getattr(env_cfg.sim, "device", self.rod_spec.device))
        self.rod_spec.__post_init__()
        env_cfg.sim.physics = newton_physics_cfg(self.rod_spec)
        # Installed here because the rod's particles have to reach the Newton
        # ModelBuilder on PhysicsEvent.MODEL_INIT, which fires while the scene
        # is being built and before the model is finalized.
        self._rod_handle = CatheterRodHandle(self.rod_spec).install()
        return env_cfg

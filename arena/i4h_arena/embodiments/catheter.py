# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Two-axis catheter embodiment backed by the i4h Warp XPBD solver."""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any, ClassVar

import isaaclab.sim as sim_utils
import numpy as np
import torch
import warp as wp
from isaaclab.assets import AssetBaseCfg
from isaaclab.managers.action_manager import ActionTerm, ActionTermCfg
from isaaclab.utils.configclass import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase

from i4h_arena.medical.catheter_drive import RouteRailedIntroducer
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
from i4h_arena.medical.route_progress import DEFAULT_INSERTION_ALLOWANCE_M, route_initial_catheter_length_m
from i4h_arena.medical.xpbd_catheter import XpbdCatheterAsset, XpbdCatheterAssetCfg
from i4h_common.navigation_route import ROUTE_SPACING_MM
from i4h_common.types import JointState


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
        self._rail = self._build_rail()

    def _build_rail(self) -> RouteRailedIntroducer | None:
        """The route rail this term feeds along, or ``None`` without a route.

        Absent a route there is nothing to prescribe the root against, so the
        term falls back to the solver's own tangent feed. That is the drive
        which folds the proximal shaft under sustained insertion -- see
        :class:`~i4h_arena.medical.catheter_drive.RouteRailedIntroducer` -- so
        it is a fallback for a rodless or route-less scene rather than a
        supported way to navigate.
        """
        route = self.cfg.route_world_m
        if route is None:
            return None
        return RouteRailedIntroducer(
            torch.as_tensor(route, dtype=torch.float32),
            self.num_envs,
            device=self.device,
        )

    @property
    def action_dim(self) -> int:
        return 3

    @property
    def tip_bend_angle(self) -> torch.Tensor:
        """Current absolute tip bend per env, in radians."""
        return self._tip_bend_angle

    @property
    def insertion_depth_m(self) -> torch.Tensor:
        """Route arc the root has been fed to, metres. Zero without a route rail."""
        if self._rail is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self._rail.depth_m

    @property
    def twist_rad(self) -> torch.Tensor:
        """Accumulated axial rotation of the root, radians. Zero without a route rail."""
        if self._rail is None:
            return torch.zeros(self.num_envs, device=self.device)
        return self._rail.twist_rad

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
        # Only the two rate terms are the drive's velocity contract; the bend is
        # a separate shape command, so it does not travel through the feed.
        self._feed(self._processed_actions[:, 0], self._processed_actions[:, 1], dt)
        limit = float(self.cfg.max_tip_bend_rad)
        # In place so the buffer the solver was handed keeps its storage.
        self._tip_bend_angle.add_(self._processed_actions[:, 2] * dt).clamp_(-limit, limit)
        self._asset.set_tip_bend(self._tip_bend_angle)

    def _feed(self, insertion_velocity: torch.Tensor, rotation_rate: torch.Tensor, dt: float) -> None:
        """Advance the wire by one step of insertion and axial rotation.

        With a route, the rail integrates the feed into arc length and the root
        is placed at that arc, so the prescribed particle stays on the vessel
        however long the operator holds insertion. The feed the rail reports as
        spent is what gets recorded, so a command the route could not take does
        not appear in the data as insertion that happened.
        """
        if self._rail is None:
            self._asset.advance(torch.stack((insertion_velocity, rotation_rate), dim=-1), dt)
            return
        spent, spin = self._rail.advance(insertion_velocity, rotation_rate, dt)
        position, quat = self._rail.root_target()
        self._asset.place_proximal(position, quat, torch.stack((spent, spin), dim=-1), dt)

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
        if self._rail is not None:
            # The rod is re-seeded from the start of the route, so the arc the
            # root is placed at has to go back with it or the wire is placed
            # mid-route against a shaft that is not there yet.
            self._rail.reset(None if env_ids is None else torch.as_tensor(env_ids, device=self.device))


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
    #: Turn of the tip polyline, so roughly a 90-degree hook at the ceiling.
    #: Matches the clamp the reference implementation settled on. This only
    #: became true once the solver's rest-curvature mapping was corrected: the
    #: earlier ``angle / n`` share realized 164 degrees here, not 86.
    max_tip_bend_rad: float = 1.5
    #: Route the wire's proximal end is fed along, as world-metre vertices, and
    #: the same polyline the rod is seeded on. Set, insertion advances the root
    #: by arc length on it instead of along the wire's own tangent, which is
    #: what keeps the prescribed particle in the vessel under sustained feed;
    #: see :class:`~i4h_arena.medical.catheter_drive.RouteRailedIntroducer`.
    #: ``None`` for a scene with no centerline, which falls back to the
    #: solver's tangent feed.
    route_world_m: tuple[tuple[float, float, float], ...] | None = None


class CArmOrbitAction(ActionTerm):
    """Rotate the complete source-detector assembly about the patient long axis."""

    cfg: CArmOrbitActionCfg

    def __init__(self, cfg: CArmOrbitActionCfg, env: Any):
        from isaaclab.sim.views import FrameView  # noqa: PLC0415

        super().__init__(cfg, env)
        # The orbit root is an authoring-only scene extra; Isaac Lab 3 gives it no
        # runtime view, so the term poses it through a FrameView over its prims.
        self._frames = FrameView(self._asset.cfg.prim_path, device=self.device)
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
        self._frames.set_local_poses(
            translations=wp.from_torch(self._root_position.contiguous(), dtype=wp.vec3f),
            orientations=wp.from_torch(self._root_orientation.contiguous(), dtype=wp.vec4f),
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


class CatheterEmbodiment(EmbodimentBase):
    """Minimal Arena embodiment whose scene entity is an external rod solver."""

    name = "catheter"
    tags: ClassVar[list[str]] = ["embodiment", "medical", "catheter"]

    def __init__(self, patient_twin_manifest: str | None = None) -> None:
        super().__init__()
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
        # Spacing comes from i4h_common rather than a literal here because
        # dataset conversion resamples the same centerline to label the goal
        # columns. Two different spacings produce two plausible routes and two
        # sets of arc figures that mean different things, with nothing raising.
        path_patient_mm, lumen_radii_mm = ordered_centerline_lumen(
            points_patient_mm,
            edges,
            target_spacing_mm=ROUTE_SPACING_MM,
            radii_mm=radii,
        )
        path_world_m = twin.patient_mm_to_world(path_patient_mm)
        path_segments = np.linalg.norm(np.diff(path_world_m, axis=0), axis=1)
        path_length = float(np.sum(path_segments))
        length = route_initial_catheter_length_m(path_length)
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
        # The default coupled solve alternates live contact with full elastic
        # blocks, including material-frame updates. The pre/post switch and
        # position-only cleanup settings remain for comparisons when contact
        # coupling is explicitly disabled with I4H_CATHETER_CONTACT_ITERATIONS=0.
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
        # The drive feeds the root along the same polyline the rod is seeded on,
        # so arc zero is where the root already sits and the first prescribed
        # pose is the one it was seeded with.
        self.action_config.catheter.route_world_m = self.rod_spec.initial_path_world_m
        # Hold the traversed shaft on the route and leave the working length
        # free. The shaft behind the tip has already been somewhere and has no
        # freedom left to spend, which is what a wire inside an introducer and
        # against vessel it threaded is like; holding it there costs the
        # operator nothing. The free window is the insertion allowance plus a
        # margin, so it always covers more than the route still to be
        # navigated and never prescribes an unmade choice.
        self.rod_spec.track_guidance = True
        self.rod_spec.track_free_distal_length_m = 1.5 * DEFAULT_INSERTION_ALLOWANCE_M
        # Both of these were attempts to damp the fold the tangent feed
        # produces, and neither worked. Guidance skips the very particle
        # insertion prescribes, so with the rail on node 1 still fell from
        # 61 mm to 15 mm of bend radius and pushed four particles through the
        # wall. Widening the baseline the feed direction is read over to 35 mm,
        # from one 4 mm segment, did not stop it either: node 1 took the bend
        # in 90% of samples across two 600-step episodes and containment stayed
        # near 25%.
        #
        # What resolved it was removing the loop instead of damping it, by
        # feeding along arc length on the route so the root's pose never
        # depends on a direction the push itself bends. See
        # :class:`~i4h_arena.medical.catheter_drive.RouteRailedIntroducer`,
        # which the keyboard drive now uses. The span below therefore only
        # reaches the solver's own tangent feed, which is what the arm drive
        # and a route-less scene still use, so it stays set for those.
        self.rod_spec.proximal_feed_span_m = 0.035
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

    def tune_env_cfg(self, env_cfg: Any) -> Any:
        env_cfg.sim.dt = 1.0 / 120.0
        # The reference catheter viewport advances controls at 30 Hz. Keep the
        # same control period while retaining 120 Hz XPBD substeps.
        env_cfg.decimation = 4
        env_cfg.sim.render_interval = 4
        # Isaac Lab 3 builds the Newton model only through physics replication:
        # without it, nothing authored in USD reaches the model next to the rod.
        env_cfg.scene.replicate_physics = True
        # Zero for the reason ``franka_catheter.py`` zeroes it: every rod is
        # seeded from the same absolute centerline, so the cloned environments
        # are coincident whatever this says. Left positive, the origins are a
        # non-zero grid under coincident rods, and ``tip_position`` -- which is
        # reported relative to the origin -- hands the policy a per-environment
        # constant of metres for one identical physical state, in a channel
        # whose real range is centimetres of travel. The RL profile cannot
        # express this: ``i4h_rl.profile`` requires a positive spacing, so the
        # override has to happen here.
        env_cfg.scene.env_spacing = 0.0

        self.rod_spec.num_envs = int(getattr(env_cfg.scene, "num_envs", 1))
        self.rod_spec.device = str(getattr(env_cfg.sim, "device", self.rod_spec.device))
        self.rod_spec.__post_init__()
        env_cfg.sim.physics = newton_physics_cfg(self.rod_spec)
        # Installed here because the rod's particles have to reach the Newton
        # ModelBuilder on PhysicsEvent.MODEL_INIT, which fires while the scene
        # is being built and before the model is finalized.
        self._rod_handle = CatheterRodHandle(self.rod_spec).install()
        return env_cfg

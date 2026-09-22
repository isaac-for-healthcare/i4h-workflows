# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The introducer the catheter is fed through, carried on an arm flange.

A clinical endovascular robot does not hold the catheter in a gripper and push
it around the room. The wire runs through a drive unit bolted to the arm, and
rollers inside that unit feed it into the patient. The arm's job is to hold the
unit on the access site; it is not the thing that advances the wire.

That makes the arm's contribution positional rather than propulsive, and this
module composes the two motions that reach the wire: the rollers feed it along
its own path, and the flange carries the whole thing rigidly.

Two rejected alternatives are worth naming, because both are easy to reach for
and each breaks something specific.

Reading insertion off the *flange's own travel* ties how much wire can be fed to
how far the arm can reach. The rod is a fixed-length 0.4 m stick and the arm has
nowhere near 0.4 m of straight-line travel from its home pose, so most of the
rod becomes unreachable. Feeding by roller keeps all of it usable.

Feeding along a *fixed introducer axis* looks right and quietly destroys
navigation. The vessel curves away from the axis, so the proximal particle
leaves the lumen: against this twin's centerline a straight rail is 11 mm off at
50 mm of depth and 44 mm off at 300 mm, well outside the vessel, and it drags
the rod off the centerline until it folds back on itself.

Feeding along the *wire's own tangent* avoids that and closes a feedback loop
instead, since the direction it reads is the one the push itself bends. What
avoids both is to feed along arc length on the route, which is what
:class:`RouteRailedIntroducer` does and what the keyboard drive uses; that
class documents the comparison and the measurements behind it.

The ops here are deliberately plain torch rather than ``isaaclab.utils.math`` so
the geometry can be tested on CPU without bringing up Isaac Sim.
"""

from __future__ import annotations

import logging
import os
import time
from dataclasses import dataclass
from typing import Any

import torch

_LOGGER = logging.getLogger(__name__)

FEED_LOG_ENV_VAR = "I4H_CATHETER_FEED"


def feed_log_seconds(environ: Any = None) -> float:
    """Seconds between roller reports, from ``I4H_CATHETER_FEED``.

    Off by default, and worth having because a stalled catheter looks the same
    on screen whichever thing stalled it. The tip-distance log says the wire is
    not moving; it cannot say whether nothing was commanded, whether the
    command was clipped, or whether the rollers hit their stop. Those have
    different fixes, so the roller reports what it was asked for, what it
    spent, and where it is in its travel.
    """
    raw = (environ if environ is not None else os.environ).get(FEED_LOG_ENV_VAR, "")
    try:
        interval = float(str(raw).strip())
    except ValueError:
        return 0.0
    return interval if interval > 0.0 else 0.0


def _quat_conjugate(quat: torch.Tensor) -> torch.Tensor:
    """Conjugate of a ``(..., 4)`` w-first quaternion."""
    return torch.cat([quat[..., :1], -quat[..., 1:]], dim=-1)


def _quat_mul(lhs: torch.Tensor, rhs: torch.Tensor) -> torch.Tensor:
    """Hamilton product of two ``(..., 4)`` w-first quaternions."""
    w1, x1, y1, z1 = lhs[..., 0], lhs[..., 1], lhs[..., 2], lhs[..., 3]
    w2, x2, y2, z2 = rhs[..., 0], rhs[..., 1], rhs[..., 2], rhs[..., 3]
    return torch.stack(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ],
        dim=-1,
    )


def twist_about_axis(
    quat_prev: torch.Tensor,
    quat_now: torch.Tensor,
    axis: torch.Tensor,
) -> torch.Tensor:
    """Signed rotation about ``axis`` carrying ``quat_prev`` to ``quat_now``.

    This is the twist half of a swing-twist split. The relative rotation is
    forced to the near hemisphere first, which is what bounds the result to
    ``[-pi, pi]``: a quaternion and its negation are the same rotation, and
    without that step the same physical motion would report either ``theta`` or
    ``theta - 2*pi`` depending on which representative the arm happened to
    produce, putting a spurious full turn into the catheter.

    Args:
        quat_prev: ``(N, 4)`` w-first quaternions from the previous step.
        quat_now: ``(N, 4)`` w-first quaternions from this step.
        axis: ``(3,)`` or ``(N, 3)`` unit axis in the same frame as the
            quaternions.

    Returns:
        ``(N,)`` signed angle in radians.
    """
    relative = _quat_mul(quat_now, _quat_conjugate(quat_prev))
    relative = torch.where(relative[..., :1] < 0.0, -relative, relative)
    axis = axis.expand_as(relative[..., 1:])
    along = torch.sum(relative[..., 1:] * axis, dim=-1)
    return 2.0 * torch.atan2(along, relative[..., 0])


def quat_about_axis(axis: torch.Tensor, angle: torch.Tensor) -> torch.Tensor:
    """``(N, 4)`` w-first quaternions rotating ``angle`` about ``axis``."""
    half = 0.5 * angle
    return torch.cat(
        [torch.cos(half).unsqueeze(-1), axis.expand(angle.shape[0], 3) * torch.sin(half).unsqueeze(-1)], dim=-1
    )


def frame_along(tangent: torch.Tensor) -> torch.Tensor:
    """``(N, 4)`` w-first frames carrying local +Z along ``tangent``.

    The rod's own convention, so a frame built here is the one the solver would
    have seeded at a node pointing this way. Nodes store local +Z along the
    tangent because the stretch constraint reaches for its neighbours along
    half-length local-Z offsets; see
    :func:`~i4h_arena.medical.catheter_initialization.rod_frames_along_polyline`,
    whose root frame is this same minimal rotation from +Z.

    Minimal rotation means the roll about the tangent is not chosen here. A
    caller that cares about roll composes its own twist on top, which is what
    axial rotation of the catheter is.
    """
    unit = tangent / torch.linalg.norm(tangent, dim=-1, keepdim=True).clamp_min(1e-12)
    # Half-angle form of the minimal rotation from +Z: the axis is +Z x t and
    # the scalar part is 1 + z.t, normalized.
    quat = torch.stack(
        [1.0 + unit[..., 2], -unit[..., 1], unit[..., 0], torch.zeros_like(unit[..., 0])],
        dim=-1,
    )
    norm = torch.linalg.norm(quat, dim=-1, keepdim=True)
    # A tangent at the -Z antipode leaves that axis undefined, the scalar part
    # vanishing with it. Any perpendicular axis is then a correct half-turn, and
    # +X is one; picking it keeps the frame finite instead of dividing by zero.
    reversed_ = torch.zeros_like(quat)
    reversed_[..., 1] = 1.0
    return torch.where(norm > 1e-6, quat / norm.clamp_min(1e-12), reversed_)


def quat_to_xyzw(quat: torch.Tensor) -> torch.Tensor:
    """Reorder ``(N, 4)`` w-first quaternions into the rod solver's xyzw."""
    return torch.cat([quat[..., 1:], quat[..., :1]], dim=-1)


def quat_to_w_first(quat: torch.Tensor) -> torch.Tensor:
    """Reorder ``(N, 4)`` xyzw quaternions from the rod solver into w-first."""
    return torch.cat([quat[..., 3:], quat[..., :3]], dim=-1)


@dataclass
class IntroducerDriveSpec:
    """Limits on what the drive unit's rollers will do.

    Attributes:
        max_insertion_velocity_mps: Clamp matching the catheter action term, so
            a fast command cannot ask for an insertion the rod cannot take.
        max_rotation_rate_radps: Same clamp for axial rotation.
        travel_limit_m: How far the wire may be fed from the access site. The
            rod is a fixed-length stick whose root is what moves, so feeding
            past its own length would drag the whole rod through the patient.
    """

    max_insertion_velocity_mps: float = 0.05
    max_rotation_rate_radps: float = 3.14159
    travel_limit_m: float = 0.36

    def __post_init__(self) -> None:
        if self.max_insertion_velocity_mps <= 0.0:
            raise ValueError("max_insertion_velocity_mps must be positive")
        if self.max_rotation_rate_radps <= 0.0:
            raise ValueError("max_rotation_rate_radps must be positive")
        if self.travel_limit_m <= 0.0:
            raise ValueError("travel_limit_m must be positive")


class LumenClamp:
    """Holds a prescribed point inside the vessel it is supposed to be inside.

    Every other particle in the rod is kept in the lumen by contact with the
    vessel wall. The proximal particle is not: it is prescribed, its inverse
    mass is zero, and so the wall constraint has nothing to push on. Nothing
    then corrects the drift it accumulates from being stepped along a chord of a
    curved path, and it was ending up 10-13 mm off the centerline against a
    median lumen radius of 8.2 mm -- outside the vessel it is meant to be
    threading.

    This restores the one constraint the prescribed point was exempt from, and
    only that one. Inside the lumen it does nothing at all, so the wire remains
    free to sit off-centre, lie against a wall, or take a bend the way contact
    left it; it acts only where the point would otherwise leave the vessel.
    Snapping the point to the centerline instead would be far stronger than the
    wall, and would rail the wire down the middle of a vessel it is supposed to
    be navigating.
    """

    def __init__(
        self,
        path_world_m: torch.Tensor,
        radii_m: torch.Tensor,
        margin_m: float,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            path_world_m: ``(S, 3)`` centerline vertices in world metres.
            radii_m: ``(S,)`` lumen radius at each vertex.
            margin_m: Kept clear of the wall, normally the catheter's own radius
                so the wire's surface rather than its axis is what touches.
            device: Where the clamp does its arithmetic.
        """
        self._device = torch.device(device)
        path = path_world_m.to(self._device, torch.float32)
        radii = radii_m.to(self._device, torch.float32).reshape(-1)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
            raise ValueError(f"path must have shape (S, 3) with S >= 2, got {tuple(path.shape)}")
        if radii.shape[0] != path.shape[0]:
            raise ValueError(f"expected one radius per path vertex, got {radii.shape[0]} for {path.shape[0]}")
        if margin_m < 0.0:
            raise ValueError(f"margin_m must be non-negative, got {margin_m}")

        self._start = path[:-1]
        self._edge = path[1:] - path[:-1]
        self._length_sq = (self._edge * self._edge).sum(-1).clamp_min(1e-12)
        # Radius is taken as the smaller of a segment's two ends. Interpolating
        # would be tighter, but erring narrow keeps the clamp from licensing a
        # point just outside a vessel that is tapering.
        self._radius = torch.minimum(radii[:-1], radii[1:])
        self._margin = float(margin_m)

    def project(self, points_world_m: torch.Tensor) -> torch.Tensor:
        """Pull points back inside the lumen, leaving interior points untouched.

        Args:
            points_world_m: ``(N, 3)`` positions in world metres.

        Returns:
            ``(N, 3)`` positions, each within its local lumen radius of the
            centerline.
        """
        query = points_world_m.to(self._device, torch.float32)

        # Closest point on every segment at once. The centerline is short
        # enough (order 100 segments) that scanning all of them beats keeping an
        # acceleration structure in step with it.
        offset = query.unsqueeze(1) - self._start.unsqueeze(0)
        alpha = ((offset * self._edge.unsqueeze(0)).sum(-1) / self._length_sq).clamp(0.0, 1.0)
        foot = self._start.unsqueeze(0) + alpha.unsqueeze(-1) * self._edge.unsqueeze(0)
        distance = torch.linalg.norm(query.unsqueeze(1) - foot, dim=-1)

        nearest = torch.argmin(distance, dim=1)
        rows = torch.arange(query.shape[0], device=self._device)
        closest = foot[rows, nearest]
        radial = distance[rows, nearest]
        allowed = (self._radius[nearest] - self._margin).clamp_min(1e-4)

        # Scaling the existing radial offset keeps the point on the same side of
        # the centerline it was already on, so a wire lying against one wall is
        # held against that wall rather than flicked across the lumen.
        scale = (allowed / radial.clamp_min(1e-9)).clamp(max=1.0)
        return closest + (query - closest) * scale.unsqueeze(-1)


class FlangeMountedIntroducer:
    """Feeds the wire along its own path while the flange carries the whole thing.

    Two motions are composed onto the rod's proximal end each step, and they
    answer different questions.

    Feed follows the wire's *current tangent*, not a fixed axis. This is the one
    that matters for whether the catheter navigates: the vessel curves, so a
    root pushed along the introducer's straight axis leaves the lumen almost
    immediately. Measured against this twin's centerline, a straight rail is
    already 11 mm off at 50 mm of depth and 44 mm off at 300 mm -- several vessel
    radii out -- and it drags the rod off the centerline until it folds back on
    itself. Following the tangent instead keeps the root on the path the wire has
    already taken, which is the path the vessel allowed.

    Transport is the flange's motion since the last step, applied rigidly. This
    is what keeps the arm in the loop: the drive unit holds the wire, so an arm
    that drifts, sags or is pushed off the site takes the wire with it. It is a
    delta rather than an absolute offset on purpose -- the wire slides *through*
    the drive unit, so the material sitting at the introducer changes as the
    rollers turn and the root legitimately travels away from the flange.
    """

    def __init__(self, spec: IntroducerDriveSpec, num_envs: int, device: str | torch.device = "cpu"):
        self._spec = spec
        self._device = torch.device(device)
        self._depth_m = torch.zeros((num_envs,), dtype=torch.float32, device=self._device)
        self._prev_flange_pos = torch.zeros((num_envs, 3), dtype=torch.float32, device=self._device)
        self._primed = torch.zeros((num_envs,), dtype=torch.bool, device=self._device)
        self._feed_logged_at = 0.0

    @property
    def spec(self) -> IntroducerDriveSpec:
        return self._spec

    @property
    def depth_m(self) -> torch.Tensor:
        """Wire fed past the access site, metres."""
        return self._depth_m

    def transport(self, flange_pos_world: torch.Tensor) -> torch.Tensor:
        """Rigid displacement the flange has imposed since the last step.

        Reports zero for an environment with no remembered pose rather than
        differencing against a placeholder, which keeps a reset from throwing
        the wire the length of the arm's jump home.

        Args:
            flange_pos_world: ``(N, 3)`` flange origin in world metres.

        Returns:
            ``(N, 3)`` displacement in world metres.
        """
        delta = flange_pos_world - self._prev_flange_pos
        delta = torch.where(self._primed.unsqueeze(-1), delta, torch.zeros_like(delta))

        self._prev_flange_pos.copy_(flange_pos_world)
        self._primed.fill_(True)
        return delta

    def advance(
        self,
        insertion_velocity: torch.Tensor,
        rotation_rate: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the rollers for one step.

        Args:
            insertion_velocity: ``(N,)`` commanded feed, metres per second.
            rotation_rate: ``(N,)`` commanded axial rotation, radians per second.
            dt: Control-step duration in seconds.

        Returns:
            The ``(insertion_velocity, rotation_rate)`` actually spent, with
            feed zeroed where the travel limit blocked it so the recorded
            command matches the wire's motion.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")
        spec = self._spec

        feed = torch.clamp(insertion_velocity, -spec.max_insertion_velocity_mps, spec.max_insertion_velocity_mps)
        spin = torch.clamp(rotation_rate, -spec.max_rotation_rate_radps, spec.max_rotation_rate_radps)

        # Clamping the depth and reporting the difference keeps the recorded
        # insertion honest at the stops: a command the rollers could not spend
        # is not insertion that happened.
        requested = self._depth_m + feed * float(dt)
        allowed = torch.clamp(requested, 0.0, spec.travel_limit_m)
        spent = (allowed - self._depth_m) / float(dt)

        self._depth_m.copy_(allowed)
        self._log_feed(insertion_velocity, feed, spent)
        return spent, spin

    def _log_feed(
        self,
        commanded: torch.Tensor,
        clamped: torch.Tensor,
        spent: torch.Tensor,
    ) -> None:
        """Report what the rollers were asked for against what they delivered.

        Three numbers, because they fail differently. ``commanded`` at zero
        means nothing asked the wire to move, which is an input question rather
        than a physics one. ``commanded`` above ``clamped`` means the velocity
        ceiling took the difference. ``clamped`` above ``spent`` means the
        travel stop did. Depth against the limit says how much wire is left.
        """
        interval = feed_log_seconds()
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._feed_logged_at < interval:
            return
        self._feed_logged_at = now
        _LOGGER.info(
            "catheter feed: t=%.2f s  commanded=%.2f mm/s  clamped=%.2f mm/s  " "spent=%.2f mm/s  depth=%.1f/%.0f mm",
            now,
            1000.0 * float(commanded[0]),
            1000.0 * float(clamped[0]),
            1000.0 * float(spent[0]),
            1000.0 * float(self._depth_m[0]),
            1000.0 * self._spec.travel_limit_m,
        )

    def root_target(
        self,
        root_pos_world: torch.Tensor,
        root_quat_world: torch.Tensor,
        tangent_world: torch.Tensor,
        transport_world: torch.Tensor,
        insertion_velocity: torch.Tensor,
        rotation_rate: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Where the wire's proximal end should be after this step.

        Args:
            root_pos_world: ``(N, 3)`` the rod's proximal particle right now.
            root_quat_world: ``(N, 4)`` its orientation, w-first.
            tangent_world: ``(N, 3)`` direction from the proximal particle to
                its neighbour. Normalized here; a degenerate first segment falls
                back to leaving the feed direction alone.
            transport_world: ``(N, 3)`` rigid displacement from :meth:`transport`.
            insertion_velocity: ``(N,)`` feed actually spent, from :meth:`advance`.
            rotation_rate: ``(N,)`` axial rotation actually spent.
            dt: Control-step duration in seconds.

        Returns:
            ``(position, quaternion)`` for the rod's root, world frame and
            w-first.
        """
        norm = torch.linalg.norm(tangent_world, dim=-1, keepdim=True)
        unit = torch.where(norm > 1e-9, tangent_world / norm.clamp_min(1e-12), torch.zeros_like(tangent_world))

        position = root_pos_world + transport_world + unit * (insertion_velocity * float(dt)).unsqueeze(-1)
        spin = quat_about_axis(unit, rotation_rate * float(dt))
        return position, _quat_mul(spin, root_quat_world)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Retract the wire and forget the flange pose.

        Dropping the remembered pose is what stops the arm's jump home from
        being transported into the wire on the next step.
        """
        if env_ids is None:
            self._depth_m.zero_()
            self._primed.zero_()
            return
        self._depth_m[env_ids] = 0.0
        self._primed[env_ids] = False


class RouteRailedIntroducer:
    """Feeds the wire by prescribing where its proximal end sits on the route.

    The proximal particle is prescribed: its inverse mass is zero, so no
    constraint in the solve moves it and whatever the drive says is where it
    goes. The only question is what the drive says, and there are three answers.

    A *fixed introducer axis* leaves the lumen as soon as the vessel curves --
    11 mm off at 50 mm of depth on this twin, 44 mm at 300 mm.

    The *wire's own tangent* fixes that and introduces a worse failure, because
    it is a feedback loop. The push direction is read from the root's first
    segment, so a root that starts to fold is fed further into the fold, which
    deepens it, which turns the direction further. Measured on ``s0011``, node
    1's bend radius fell from 61 mm at reset to 15 mm over 400 steps of
    pushing, four particles ended up through the vessel wall, and the shaft
    buckled instead of advancing the tip. Widening the baseline the direction
    is measured over and railing the traversed shaft onto the path both damped
    it and neither stopped it, because both leave the root itself free to go
    wherever the accumulated direction points.

    This class takes the third answer: the root's *arc length along the route*
    is the state, and its pose is looked up from that. Nothing accumulates, so
    there is no loop to close -- an inextensible shaft advancing a millimetre
    at the tip advances a millimetre at the root, which makes arc length the
    honest integral of the feed, and the root is on the route by construction
    at every value of it.

    That is also what the hardware does. The wire at the access site runs
    through an introducer sheath and then along vessel it has already
    traversed, so its proximal end has no lateral freedom to fold with; the
    freedom the operator has is all distal. Prescribing the root on the route
    reproduces that constraint rather than modelling the sheath.

    What stays free is everything the episode is about. Only node 0 is
    prescribed, and only along a route the wire has already been fed through:
    the shaft's shape, its contact with the wall, the steerable tip and every
    millimetre ahead of the tip are left to the solve and the operator.
    """

    def __init__(
        self,
        path_world_m: torch.Tensor,
        num_envs: int,
        device: str | torch.device = "cpu",
    ):
        """
        Args:
            path_world_m: ``(S, 3)`` route vertices in world metres, the same
                polyline the rod is seeded along. Arc zero is its first vertex,
                which is where the root starts.
            num_envs: Environments fed in parallel.
            device: Where the lookup runs. Kept on the simulation device so
                feeding the wire does not synchronize on the host each step.
        """
        self._device = torch.device(device)
        path = path_world_m.to(self._device, torch.float32)
        if path.ndim != 2 or path.shape[0] < 2 or path.shape[1] != 3:
            raise ValueError(f"route must have shape (S, 3) with S >= 2, got {tuple(path.shape)}")
        if not torch.isfinite(path).all():
            raise ValueError("route contains non-finite vertices")

        span = path[1:] - path[:-1]
        length = torch.linalg.norm(span, dim=-1)
        if bool((length <= 0.0).any()):
            raise ValueError("consecutive route vertices must be distinct")
        self._start = path[:-1]
        self._direction = span / length.unsqueeze(-1)
        # Arc at each segment's start, so a lookup interpolates from there.
        self._arc_at_start = torch.cat([torch.zeros(1, device=self._device), torch.cumsum(length, dim=0)[:-1]])
        self._route_length_m = float(length.sum())

        self._arc_m = torch.zeros((num_envs,), dtype=torch.float32, device=self._device)
        self._twist_rad = torch.zeros((num_envs,), dtype=torch.float32, device=self._device)
        self._feed_logged_at = 0.0

    @property
    def route_length_m(self) -> float:
        """Arc length of the whole route, metres."""
        return self._route_length_m

    @property
    def depth_m(self) -> torch.Tensor:
        """Route arc the root has advanced along, metres."""
        return self._arc_m

    def advance(
        self,
        insertion_velocity: torch.Tensor,
        rotation_rate: torch.Tensor,
        dt: float,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Run the rollers for one step.

        Args:
            insertion_velocity: ``(N,)`` commanded feed, metres per second.
            rotation_rate: ``(N,)`` commanded axial rotation, radians per second.
            dt: Control-step duration in seconds.

        Returns:
            The ``(insertion_velocity, rotation_rate)`` actually spent. Feed is
            reduced where the route ran out, so the recorded command matches
            the wire's motion rather than the keypress.
        """
        if dt <= 0.0:
            raise ValueError(f"dt must be positive, got {dt}")

        # The end of the route is the stop, and it is the honest one: past it
        # there is no vessel for the root to be on, so there is no pose to
        # prescribe. Useful travel runs out earlier anyway, when the tip
        # reaches the far end and the shaft's own length is what's left.
        requested = self._arc_m + insertion_velocity * float(dt)
        allowed = torch.clamp(requested, 0.0, self._route_length_m)
        spent = (allowed - self._arc_m) / float(dt)
        self._arc_m.copy_(allowed)
        self._twist_rad.add_(rotation_rate * float(dt))

        self._log_feed(insertion_velocity, spent)
        return spent, rotation_rate

    def root_target(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Where the wire's proximal end should be, from the arc it has reached.

        Takes no current rod state, which is the whole point: the pose is a
        function of the feed integral and the route, so it cannot inherit an
        error from where the root drifted to last step.

        Returns:
            ``(position, quaternion)`` for the rod's root, world frame and
            w-first, matching what
            :meth:`~i4h_arena.medical.xpbd_catheter.XpbdCatheterAsset.place_proximal`
            consumes.
        """
        # ``searchsorted`` on the segment starts gives the segment containing
        # each arc; an arc landing exactly on the far end belongs to the last.
        index = torch.clamp(
            torch.searchsorted(self._arc_at_start, self._arc_m.contiguous(), right=True) - 1,
            0,
            self._start.shape[0] - 1,
        )
        direction = self._direction[index]
        position = self._start[index] + direction * (self._arc_m - self._arc_at_start[index]).unsqueeze(-1)
        # Axial rotation is composed on here rather than sent as a rate command,
        # because the solver latches one proximal command per step and a pose
        # would overwrite the rate. See
        # :meth:`catheter_vasculature_solver.xpbd_rod_solver.CathRodSolver.set_root_pose_gpu`.
        spin = quat_about_axis(direction, self._twist_rad)
        return position, _quat_mul(spin, frame_along(direction))

    def _log_feed(self, commanded: torch.Tensor, spent: torch.Tensor) -> None:
        """Report what the rollers were asked for against what the route allowed."""
        interval = feed_log_seconds()
        if interval <= 0.0:
            return
        now = time.monotonic()
        if now - self._feed_logged_at < interval:
            return
        self._feed_logged_at = now
        _LOGGER.info(
            "catheter feed: t=%.2f s  commanded=%.2f mm/s  spent=%.2f mm/s  route=%.1f/%.0f mm",
            now,
            1000.0 * float(commanded[0]),
            1000.0 * float(spent[0]),
            1000.0 * float(self._arc_m[0]),
            1000.0 * self._route_length_m,
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Return the root to the start of the route, untwisted."""
        if env_ids is None:
            self._arc_m.zero_()
            self._twist_rad.zero_()
            return
        self._arc_m[env_ids] = 0.0
        self._twist_rad[env_ids] = 0.0


__all__ = [
    "FlangeMountedIntroducer",
    "LumenClamp",
    "RouteRailedIntroducer",
    "IntroducerDriveSpec",
    "frame_along",
    "quat_about_axis",
    "quat_to_w_first",
    "quat_to_xyzw",
    "twist_about_axis",
]

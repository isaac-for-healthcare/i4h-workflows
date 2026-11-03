# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deforming, contrast-filled vessel's contribution to a projection's line integral.

Counterpart to :mod:`~i4h_arena.medical.catheter_attenuation`, and the other half of the carve:
:mod:`~i4h_arena.medical.vessel_carve` takes the contrast lumen out of the static volume, and
this puts it back every frame at wherever the solver has moved the centerline to and whatever
concentration :mod:`~i4h_arena.medical.contrast_bolus` says is there. Because both the carve and
the compositing describe the same tube, the result at rest reproduces the original scan, and
the difference from rest is exactly the wall motion and the bolus.

Beer-Lambert makes this composable without re-rendering the volume. Transmission is
:math:`\\exp(-\\int \\mu \\, ds)`, so attenuators add in the exponent: the vessel's chord
integral is accumulated separately and multiplied into the volume's transmission, the same way
the catheter already is. Nothing about the anatomy volume has to be touched per frame.

**Why chords and not samples.** The lumen is 6-30 mm across and the wall moves 0.5-2 mm. Ray
marching at a step fine enough to resolve that costs hundreds of samples per pixel and still
quantises the boundary at the step size, which aliases into exactly the signal being measured.
A frustum is convex, so a ray crosses it once and the entry and exit parameters come from one
quadratic -- the geometry is resolved exactly, at a cost independent of the vessel's size.

**Why frusta and not cylinders.** The segment primitive has to match the one the carve removed,
or the two do not cancel at rest and the residual appears as a static vessel-shaped artifact.
The carve is a sphere sweep, whose radius interpolates linearly between nodes; the matching
analytic primitive is therefore the truncated cone, not the cylinder. On ``s0011`` the radius
changes by a median 0.34 mm and up to 2.14 mm across a single 1.8 mm segment, so holding it
constant would misplace the wall by up to 1.07 mm -- the size of the effect being rendered.
The cost is that a taper this steep (a tangent up to 0.82, so a 39 degree half-angle) puts rays
inside the cone's own half-angle, where the quadratic's leading coefficient goes negative and
containment is the region *outside* the roots rather than between them. That branch is real for
near-axial rays and is handled below rather than clamped away.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

from .catheter_attenuation import _detector_rays

if TYPE_CHECKING:
    import torch

    from .projection import ProjectionGeometry

#: Segments shorter than this carry no geometry worth a chord, and a degenerate axis makes the
#: frustum's axial frame ill-conditioned. Matches the catheter kernel's threshold.
_MIN_SEGMENT_MM = 1.0e-4


@wp.func
def _clip_halfspace(low: float, high: float, constant: float, rate: float) -> wp.vec2:
    """Narrow ``[low, high]`` to the ray parameters satisfying ``constant + rate * t >= 0``.

    An empty result comes back as a reversed interval, which stays empty under any further
    clipping, so the caller tests once at the end rather than after every plane.
    """
    if rate > 1.0e-12:
        return wp.vec2(wp.max(low, -constant / rate), high)
    if rate < -1.0e-12:
        return wp.vec2(low, wp.min(high, -constant / rate))
    if constant < 0.0:
        return wp.vec2(1.0, -1.0)
    return wp.vec2(low, high)


@wp.func
def _frustum_chord(
    origin: wp.vec3,
    direction: wp.vec3,
    start: wp.vec3,
    end: wp.vec3,
    radius_start: float,
    radius_end: float,
    normal_start: wp.vec3,
    normal_end: wp.vec3,
    span: float,
) -> wp.vec2:
    """Chord length through one truncated cone, and the axial coordinate of its midpoint.

    The midpoint comes back because attenuation varies linearly along the segment, so the mean
    of a linear function over the chord is its value at the chord's centre -- exact, and one
    evaluation instead of a quadrature.

    Returns:
        ``(chord_length, axial_fraction)``, where the fraction is the chord midpoint's position
        along the segment in ``[0, 1]``. A length of zero means the ray misses.
    """
    axis_vector = end - start
    length = wp.length(axis_vector)
    if length <= _MIN_SEGMENT_MM:
        return wp.vec2(0.0, 0.0)

    axis = axis_vector / length
    offset = origin - start
    axial_offset = wp.dot(offset, axis)
    axial_direction = wp.dot(direction, axis)
    radial_offset = offset - axis * axial_offset
    radial_direction = direction - axis * axial_direction

    taper = (radius_end - radius_start) / length
    radius_at_origin = radius_start + taper * axial_offset
    radius_rate = taper * axial_direction

    # Clip to the joint planes first. Each is shared with the neighbouring segment, so the sign
    # has to be chosen from this segment's own axis to keep its own body: the neighbour makes the
    # opposite choice at the same node and the two half-spaces partition the joint exactly.
    plane_start = normal_start * wp.sign(wp.dot(normal_start, axis))
    plane_end = normal_end * wp.sign(wp.dot(normal_end, axis))
    bounds = _clip_halfspace(0.0, span, wp.dot(offset, plane_start), wp.dot(direction, plane_start))
    bounds = _clip_halfspace(
        bounds[0], bounds[1], wp.dot(end - origin, plane_end), -wp.dot(direction, plane_end)
    )
    # A mitred plane leans past its node, far enough in a tight turn that the linear radius can
    # cross zero into the mirror cone on the far side of the apex. Clipping to where the radius
    # is still positive removes that without assuming the plane stays near the end of the axis.
    bounds = _clip_halfspace(bounds[0], bounds[1], radius_at_origin, radius_rate)
    low = bounds[0]
    high = bounds[1]
    if high <= low:
        return wp.vec2(0.0, 0.0)

    # Containment is |radial(t)| <= R(t) with R linear in t, which squares to a quadratic.
    quadratic = wp.dot(radial_direction, radial_direction) - radius_rate * radius_rate
    linear = 2.0 * (wp.dot(radial_direction, radial_offset) - radius_at_origin * radius_rate)
    constant = wp.dot(radial_offset, radial_offset) - radius_at_origin * radius_at_origin

    entry = float(0.0)
    exit_ = float(-1.0)
    if wp.abs(quadratic) > 1.0e-12:
        discriminant = linear * linear - 4.0 * quadratic * constant
        if discriminant >= 0.0:
            root = wp.sqrt(discriminant)
            first = (-linear - root) / (2.0 * quadratic)
            second = (-linear + root) / (2.0 * quadratic)
            lower = wp.min(first, second)
            upper = wp.max(first, second)
            if quadratic > 0.0:
                entry = wp.max(low, lower)
                exit_ = wp.min(high, upper)
            else:
                # The ray lies inside the cone's own half-angle, so containment holds outside the
                # roots. A truncated cone is convex, so after the clipping above at most one
                # of the two tails survives; taking whichever is non-empty keeps that exact without
                # assuming which one it is.
                entry = low
                exit_ = wp.min(high, lower)
                if exit_ <= entry:
                    entry = wp.max(low, upper)
                    exit_ = high
        elif quadratic < 0.0:
            # No crossing and containment holds outside the roots, so the whole clipped span is
            # inside: an axial ray threading the cone from end plane to end plane.
            entry = low
            exit_ = high
    elif wp.abs(linear) > 1.0e-12:
        crossing = -constant / linear
        if linear > 0.0:
            entry = low
            exit_ = wp.min(high, crossing)
        else:
            entry = wp.max(low, crossing)
            exit_ = high
    elif constant <= 0.0:
        entry = low
        exit_ = high

    if exit_ <= entry:
        return wp.vec2(0.0, 0.0)
    midpoint_axial = axial_offset + 0.5 * (entry + exit_) * axial_direction
    return wp.vec2(exit_ - entry, wp.clamp(midpoint_axial / length, 0.0, 1.0))


@wp.kernel
def _accumulate_vessel_line_integral(
    source: wp.vec3,
    first_pixel: wp.vec3,
    column_step: wp.vec3,
    row_step: wp.vec3,
    starts: wp.array2d(dtype=wp.vec3),
    ends: wp.array2d(dtype=wp.vec3),
    radius_start: wp.array2d(dtype=float),
    radius_end: wp.array2d(dtype=float),
    normal_start: wp.array2d(dtype=wp.vec3),
    normal_end: wp.array2d(dtype=wp.vec3),
    mu_start: wp.array2d(dtype=float),
    mu_end: wp.array2d(dtype=float),
    line_integral: wp.array3d(dtype=float),
) -> None:
    env, row, column = wp.tid()
    beam = first_pixel + column_step * float(column) + row_step * float(row) - source
    span = wp.length(beam)
    total = float(0.0)
    if span > 1.0e-6:
        direction = beam / span
        for index in range(starts.shape[1]):
            result = _frustum_chord(
                source,
                direction,
                starts[env, index],
                ends[env, index],
                radius_start[env, index],
                radius_end[env, index],
                normal_start[env, index],
                normal_end[env, index],
                span,
            )
            if result[0] > 0.0:
                # Linear in the axial coordinate, so the chord's mean attenuation is its value at
                # the chord midpoint. Contrast concentration varies node to node, and this is what
                # makes a bolus front render as a gradient rather than a staircase.
                mu = mu_start[env, index] + (mu_end[env, index] - mu_start[env, index]) * result[1]
                total += mu * result[0]
    line_integral[env, row, column] = total


@dataclass(frozen=True, slots=True)
class VesselState:
    """One frame's vessel geometry and contrast, per environment.

    Args:
        points_mm: Node positions in volume millimetres, ``(envs, nodes, 3)``. These come from the
            deformation solver, so they move; everything else here is indexed to match.
        edges: Node index pairs, ``(segments, 2)``, shared across environments because the
            topology is the anatomy's and does not deform.
        radii_mm: Lumen radius per node, ``(envs, nodes)`` or ``(nodes,)`` when undeformed.
        mu_per_mm: Attenuation the contrast adds over the carve's fill, per node, ``(envs,
            nodes)`` or ``(nodes,)``. Supplied by the transport model.
    """

    points_mm: np.ndarray
    edges: np.ndarray
    radii_mm: np.ndarray
    mu_per_mm: np.ndarray

    def __post_init__(self) -> None:
        points = np.asarray(self.points_mm, dtype=np.float32)
        if points.ndim != 3 or points.shape[-1] != 3:
            raise ValueError(f"points_mm must be (envs, nodes, 3), got {points.shape}")
        envs, nodes = points.shape[0], points.shape[1]
        edges = np.asarray(self.edges, dtype=np.int32).reshape(-1, 2)
        if edges.size and (edges.min() < 0 or edges.max() >= nodes):
            raise ValueError(f"edges index outside the {nodes} supplied nodes")
        object.__setattr__(self, "points_mm", points)
        object.__setattr__(self, "edges", edges)
        for name in ("radii_mm", "mu_per_mm"):
            values = np.asarray(getattr(self, name), dtype=np.float32)
            # Length-checked before broadcasting, so a wrong-sized array reports which field is
            # wrong rather than letting numpy's remapped-shape message surface instead.
            if values.ndim == 1 and len(values) == nodes:
                values = np.broadcast_to(values, (envs, nodes))
            if values.shape != (envs, nodes):
                raise ValueError(f"{name} must be ({envs}, {nodes}) or ({nodes},), got {values.shape}")
            object.__setattr__(self, name, np.ascontiguousarray(values))


class VesselAttenuation:
    """The contrast-filled lumen's contribution to a projection's line integral.

    Deliberately shaped like :class:`~i4h_arena.medical.catheter_attenuation.CatheterAttenuation`:
    both are analytic chord integrators that compose into the renderer's transmission, and keeping
    one pattern means the detector ray construction is shared rather than reimplemented, which is
    where a registration error between the composited geometry and the anatomy would come from.
    """

    def __init__(self, *, device: str | None = None) -> None:
        self._device = wp.get_device(device)
        self._line_integral: wp.array | None = None
        self._buffers: dict[str, wp.array] = {}

    def line_integral(
        self,
        state: VesselState,
        projection: ProjectionGeometry,
        *,
        width: int,
        height: int,
        as_torch: bool = False,
    ) -> np.ndarray | torch.Tensor:
        """Integrate the vessel's attenuation along every detector ray.

        Args:
            state: This frame's node positions, radii and contrast attenuation.
            projection: The C-arm pose, in the renderer's own volume frame.
            width: Detector columns.
            height: Detector rows.
            as_torch: Return a torch tensor on the renderer's device rather than numpy.

        Returns:
            ``(envs, height, width)`` of :math:`\\int \\mu \\, ds` in millimetres, ready to be
            added to any other attenuator's integral before exponentiating.
        """
        envs = state.points_mm.shape[0]
        buffer = self._output_buffer(envs=envs, height=height, width=width)
        buffer.zero_()

        segments = _segment_arrays(state)
        starts, ends, radius_start, radius_end, normal_start, normal_end, mu_start, mu_end = segments
        if starts.shape[1] == 0:
            return self._read_out(buffer, as_torch=as_torch)

        source, first_pixel, column_step, row_step = _detector_rays(projection, width=width, height=height)
        stream = wp.stream_from_torch(wp.device_to_torch(self._device)) if as_torch and self._device.is_cuda else None
        with wp.ScopedStream(stream):
            wp.launch(
                _accumulate_vessel_line_integral,
                dim=(envs, height, width),
                inputs=[
                    source,
                    first_pixel,
                    column_step,
                    row_step,
                    self._upload("starts", starts, wp.vec3),
                    self._upload("ends", ends, wp.vec3),
                    self._upload("radius_start", radius_start, float),
                    self._upload("radius_end", radius_end, float),
                    self._upload("normal_start", normal_start, wp.vec3),
                    self._upload("normal_end", normal_end, wp.vec3),
                    self._upload("mu_start", mu_start, float),
                    self._upload("mu_end", mu_end, float),
                    buffer,
                ],
                device=self._device,
            )
        return self._read_out(buffer, as_torch=as_torch)

    def _upload(self, name: str, values: np.ndarray, dtype: object) -> wp.array:
        """Reuse one device array per input, so a cine run does not reallocate every frame.

        The catheter integrator allocates fresh arrays per call, which is affordable for a
        hundred segments and is not for a vessel tree evaluated at every control step.
        """
        existing = self._buffers.get(name)
        if existing is None or existing.shape != values.shape[: existing.ndim] or existing.dtype is not dtype:
            self._buffers[name] = wp.array(values, dtype=dtype, device=self._device)
            return self._buffers[name]
        existing.assign(values)
        return existing

    def _output_buffer(self, *, envs: int, height: int, width: int) -> wp.array:
        if self._line_integral is None or self._line_integral.shape != (envs, height, width):
            self._line_integral = wp.zeros((envs, height, width), dtype=float, device=self._device)
        return self._line_integral

    def _read_out(self, buffer: wp.array, *, as_torch: bool) -> np.ndarray | torch.Tensor:
        # Copied out because the buffer is reused next frame, and on a CPU device `numpy()` is a
        # view onto it rather than a snapshot.
        if as_torch:
            return wp.to_torch(buffer).clone()
        return buffer.numpy().copy()


def _joint_normals(points: np.ndarray, edges: np.ndarray, unit: np.ndarray) -> np.ndarray:
    """One plane direction per node, shared by every segment meeting there.

    Clipping each frustum to a plane perpendicular to its own axis leaves a wedge uncovered on
    the outside of every turn and a double-counted overlap on the inside, because the two planes
    at a shared node are not the same plane. The centerline follows a voxel staircase, so on
    ``s0011`` it turns a mean 32 degrees per node, which on a 7.5 mm radius displaces the joint
    by 3.7 mm either way -- twice the 1.8 mm segment length. Summing the per-segment chords then
    alternately counts the lumen twice and misses it, and the vessel renders as a barcode.
    Mitring the joint the way adjoining pipes are gives both segments one plane, so the chain
    partitions space and the chords really are additive.

    The direction is deliberately unsigned: each segment orients it from its own axis in the
    kernel. Centerline edges are not consistently oriented along flow -- ``s0011`` has six nodes
    that two edges both point into -- so a normal whose sign came from the stored edge direction
    would cancel at exactly those nodes instead of bisecting.

    A node where more than two segments meet has no plane separating all of them. The average is
    still shared, so a parent does not overlap its children, but the children overlap each other
    within a radius of the bifurcation.
    """
    # Any one incident direction, as the reference each node's contributions are aligned to.
    # Which one is arbitrary and the bisector does not depend on it, only its overall sign does.
    reference = np.zeros_like(points)
    for side in (edges[:, 0], edges[:, 1]):
        reference[:, side] = unit

    normals = np.zeros_like(points)
    for side in (edges[:, 0], edges[:, 1]):
        aligned = unit * np.sign(np.sum(unit * reference[:, side], axis=-1, keepdims=True))
        np.add.at(normals, (slice(None), side), aligned)

    length = np.linalg.norm(normals, axis=-1, keepdims=True)
    degenerate = length <= 1.0e-6
    return np.where(degenerate, reference, normals / np.where(degenerate, 1.0, length))


def _segment_arrays(state: VesselState) -> tuple[np.ndarray, ...]:
    """Per-segment endpoints, radii and attenuation, with degenerate segments dropped.

    Degenerate segments are dropped on the geometry of the *first* environment so that every
    environment keeps the same segment count and the launch stays one rectangular grid. Nodes
    collapse when the solver pinches a vessel, and that happens per environment; using a shared
    mask means such a segment still contributes a near-zero chord elsewhere rather than
    desynchronising the batch.
    """
    points = state.points_mm
    edges = state.edges
    if edges.size == 0:
        vectors = np.zeros((points.shape[0], 0, 3), dtype=np.float32)
        empty = np.zeros((points.shape[0], 0), dtype=np.float32)
        return (vectors,) * 2 + (empty,) * 2 + (vectors,) * 2 + (empty,) * 2

    left, right = edges[:, 0], edges[:, 1]
    starts = points[:, left]
    ends = points[:, right]
    axis = ends - starts
    length = np.linalg.norm(axis, axis=-1, keepdims=True)
    unit = np.divide(axis, length, out=np.zeros_like(axis), where=length > _MIN_SEGMENT_MM)
    normals = _joint_normals(points, edges, unit)

    keep = length[0, :, 0] > _MIN_SEGMENT_MM
    return (
        np.ascontiguousarray(starts[:, keep]),
        np.ascontiguousarray(ends[:, keep]),
        np.ascontiguousarray(state.radii_mm[:, left][:, keep]),
        np.ascontiguousarray(state.radii_mm[:, right][:, keep]),
        np.ascontiguousarray(normals[:, left][:, keep]),
        np.ascontiguousarray(normals[:, right][:, keep]),
        np.ascontiguousarray(state.mu_per_mm[:, left][:, keep]),
        np.ascontiguousarray(state.mu_per_mm[:, right][:, keep]),
    )


__all__ = ["VesselAttenuation", "VesselState"]

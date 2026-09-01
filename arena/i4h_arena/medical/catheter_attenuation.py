# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Compositing the catheter into the beam as attenuating geometry.

The catheter used to be drawn onto the finished image as a dark polyline. That put it in front of
everything: it stayed equally dark behind the spine, it never grew as it approached the detector,
and it never reached the ``attenuation`` channel a policy reads.

Here each span between two nodes is a cylinder carrying its own linear attenuation coefficient, and
its contribution ``sum(mu_i * chord_i)`` is added to the line integral the volume already produced.
Beer-Lambert is additive in the exponent, so this is the fused ray march the interactive catheter
viewport performs inside its shader, written against the pinned renderer's detector grid instead:
occlusion by dense anatomy, cone-beam magnification and the detector's blur then all follow from
the geometry rather than being painted on.

Two details differ from a literal reading of that shader. Chords are solved analytically rather
than sampled, because a half-millimetre shaft is thinner than a sensible march step and sampling
aliases it in and out of view along its length. And spans end flat rather than rounded, so that
consecutive spans meet at a shared plane instead of overlapping: solvers place nodes a fraction of
a millimetre apart, and rounded ends would then have several spans claiming the same material and
count its attenuation once per span. What that leaves unmodelled is a wedge on the outside of each
bend, roughly the radius times the half-angle, well under a pixel for a catheter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np
import warp as wp

if TYPE_CHECKING:
    from .slang_fluoroscopy import ProjectionGeometry

# Below this a span carries no geometry: its endpoints coincide and its axis is undefined.
_MIN_SEGMENT_MM = 1.0e-6


@dataclass(frozen=True, slots=True)
class CatheterMaterial:
    """Attenuation of the catheter shaft and of the radiopaque band at its tip.

    The defaults describe a braided nitinol shaft ending in a tungsten-loaded marker band, using
    the same coefficients as the viewport's segment table: nitinol around 0.8, tungsten around 3.0
    and platinum around 5.0 per millimetre. A marker is what makes the tip readable on a live
    image, so it is worth carrying separately even at this level of detail — a uniform shaft gives
    an operator no cue for where the tip actually is.
    """

    shaft_mu_per_mm: float = 0.8
    tip_mu_per_mm: float = 3.0
    tip_length_mm: float = 2.0

    def __post_init__(self) -> None:
        for name in ("shaft_mu_per_mm", "tip_mu_per_mm", "tip_length_mm"):
            value = float(getattr(self, name))
            if not np.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative, got {value}")

    def segment_mu_per_mm(self, segment_lengths_mm: np.ndarray) -> np.ndarray:
        """Assign a coefficient to every span, marking the distal end radiopaque.

        Node order runs from the base to the tip, matching the guidance overlay's choice of the
        last valid node as the tip, so the marker occupies the final ``tip_length_mm`` of the
        polyline rather than a fixed number of spans, whose length depends on the solver.
        """
        lengths = np.asarray(segment_lengths_mm, dtype=np.float64)
        distance_to_tip_mm = np.cumsum(lengths[::-1])[::-1] - 0.5 * lengths
        return np.where(
            distance_to_tip_mm <= self.tip_length_mm,
            self.tip_mu_per_mm,
            self.shaft_mu_per_mm,
        ).astype(np.float32)


@wp.func
def _cylinder_chord(
    origin: wp.vec3,
    direction: wp.vec3,
    start: wp.vec3,
    end: wp.vec3,
    radius: float,
    span: float,
) -> float:
    """Length of the ray lying inside one span, in the units of the endpoints."""
    axis = end - start
    length = wp.length(axis)
    chord = float(0.0)
    if length > _MIN_SEGMENT_MM:
        axis = axis / length
        offset = origin - start
        axial_direction = wp.dot(direction, axis)
        axial_offset = wp.dot(offset, axis)
        radial_direction = direction - axis * axial_direction
        radial_offset = offset - axis * axial_offset

        # Within `radius` of the axis, from the quadratic in the plane normal to it.
        quadratic = wp.dot(radial_direction, radial_direction)
        radial = wp.dot(radial_offset, radial_offset) - radius * radius
        entry = float(0.0)
        exit_ = float(-1.0)
        if quadratic > 1.0e-12:
            linear = 2.0 * wp.dot(radial_direction, radial_offset)
            discriminant = linear * linear - 4.0 * quadratic * radial
            if discriminant > 0.0:
                root = wp.sqrt(discriminant)
                entry = (-linear - root) / (2.0 * quadratic)
                exit_ = (-linear + root) / (2.0 * quadratic)
        elif radial < 0.0:
            # The ray runs along the axis inside the shaft, so only the end planes bound it.
            entry = -1.0e30
            exit_ = 1.0e30

        # Between the two end planes, which is what keeps neighbouring spans from overlapping.
        if exit_ > entry:
            if wp.abs(axial_direction) > 1.0e-12:
                first = -axial_offset / axial_direction
                second = (length - axial_offset) / axial_direction
                entry = wp.max(entry, wp.min(first, second))
                exit_ = wp.min(exit_, wp.max(first, second))
            elif axial_offset < 0.0 or axial_offset > length:
                exit_ = entry - 1.0

        # A cylinder is convex, so this is one contiguous interval; keep the part of it that lies
        # on the beam between the source and the detector.
        entry = wp.max(entry, 0.0)
        exit_ = wp.min(exit_, span)
        if exit_ > entry:
            chord = exit_ - entry
    return chord


@wp.kernel
def _accumulate_catheter_line_integral(
    source: wp.vec3,
    first_pixel: wp.vec3,
    column_step: wp.vec3,
    row_step: wp.vec3,
    starts: wp.array(dtype=wp.vec3),
    ends: wp.array(dtype=wp.vec3),
    mu_per_mm: wp.array(dtype=float),
    radius_mm: float,
    line_integral: wp.array2d(dtype=float),
) -> None:
    row, column = wp.tid()
    beam = first_pixel + column_step * float(column) + row_step * float(row) - source
    span = wp.length(beam)
    total = float(0.0)
    if span > 1.0e-6:
        direction = beam / span
        for index in range(starts.shape[0]):
            total += mu_per_mm[index] * _cylinder_chord(source, direction, starts[index], ends[index], radius_mm, span)
    line_integral[row, column] = total


def _vec3(values: np.ndarray) -> wp.vec3:
    return wp.vec3(float(values[0]), float(values[1]), float(values[2]))


class CatheterAttenuation:
    """The catheter's own contribution to a projection's line integral."""

    def __init__(self, material: CatheterMaterial | None = None, *, device: str | None = None) -> None:
        self._material = material if material is not None else CatheterMaterial()
        self._device = wp.get_device(device)
        self._line_integral: wp.array | None = None

    @property
    def material(self) -> CatheterMaterial:
        return self._material

    def line_integral(
        self,
        points_volume_mm: np.ndarray,
        radius_mm: float,
        projection: ProjectionGeometry,
        *,
        width: int,
        height: int,
    ) -> np.ndarray:
        """Return ``sum(mu_i * chord_i)`` per pixel, ready to add to the volume's line integral.

        ``points_volume_mm`` holds the active nodes in the renderer's volume millimetre frame,
        ordered base to tip.
        """
        points = np.asarray(points_volume_mm, dtype=np.float32)
        if points.ndim != 2 or points.shape[-1] != 3:
            raise ValueError("points_volume_mm must have shape (num_nodes, 3)")
        if not np.isfinite(points).all():
            raise ValueError("points_volume_mm must contain only finite values")
        radius = float(radius_mm)
        if not np.isfinite(radius) or radius <= 0.0:
            raise ValueError(f"radius_mm must be positive and finite, got {radius_mm}")

        buffer = self._line_integral_buffer(height=height, width=width)
        starts = points[:-1]
        ends = points[1:]
        lengths_mm = np.linalg.norm(ends - starts, axis=-1)
        carries_geometry = lengths_mm > _MIN_SEGMENT_MM
        if not carries_geometry.any():
            return np.zeros((height, width), dtype=np.float32)
        mu_per_mm = self._material.segment_mu_per_mm(lengths_mm)[carries_geometry]

        source, first_pixel, column_step, row_step = _detector_rays(projection, width=width, height=height)
        wp.launch(
            _accumulate_catheter_line_integral,
            dim=(height, width),
            inputs=[
                source,
                first_pixel,
                column_step,
                row_step,
                wp.array(starts[carries_geometry], dtype=wp.vec3, device=self._device),
                wp.array(ends[carries_geometry], dtype=wp.vec3, device=self._device),
                wp.array(mu_per_mm, dtype=float, device=self._device),
                radius,
                buffer,
            ],
            device=self._device,
        )
        # Copied out because the buffer is about to be overwritten by the next frame, and on a CPU
        # device `numpy()` is a view onto it rather than a snapshot.
        return buffer.numpy().copy()

    def _line_integral_buffer(self, *, height: int, width: int) -> wp.array:
        """One detector-sized buffer per sensor, reused so a cine run does not churn allocations."""
        if self._line_integral is None or self._line_integral.shape != (height, width):
            self._line_integral = wp.zeros((height, width), dtype=float, device=self._device)
        return self._line_integral


def _detector_rays(
    projection: ProjectionGeometry,
    *,
    width: int,
    height: int,
) -> tuple[wp.vec3, wp.vec3, wp.vec3, wp.vec3]:
    """Source, first pixel centre and pixel steps in volume millimetres.

    The renderer's kernel puts pixel ``(column, row)`` at ``(index + 0.5 - count / 2) * pitch`` in
    detector coordinates and its source at ``-sid`` along the beam axis. Both are reproduced here,
    half-pixel offset included, so the catheter occludes the anatomy it is registered against.
    """
    axes = np.asarray(projection.local_to_volume, dtype=np.float64)
    horizontal, vertical, beam = axes[:, 0], axes[:, 1], axes[:, 2]
    isocenter_mm = np.asarray(projection.isocenter_volume_mm, dtype=np.float64)
    pitch_mm = float(projection.pixel_spacing_mm)
    source_mm = isocenter_mm - projection.source_to_isocenter_mm * beam
    detector_mm = isocenter_mm + (projection.source_to_detector_mm - projection.source_to_isocenter_mm) * beam
    first_pixel_mm = (
        detector_mm + horizontal * ((0.5 - 0.5 * width) * pitch_mm) + vertical * ((0.5 - 0.5 * height) * pitch_mm)
    )
    return _vec3(source_mm), _vec3(first_pixel_mm), _vec3(horizontal * pitch_mm), _vec3(vertical * pitch_mm)

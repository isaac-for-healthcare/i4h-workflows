# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Tests for building a deformable vessel from a patient twin manifest.

The unit conversion is the whole point of these: the twin stores its centerline
in patient millimetres and the solver wants Isaac world metres, and a silent
factor of a thousand would put the lumen nowhere near the anatomy.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from i4h_arena.medical.vessel_deformation import (
    VESSEL_ANGULAR_DAMPING,
    VESSEL_ENDPOINTS_LOCKED,
    VESSEL_LINEAR_DAMPING,
    VESSEL_RESPONSE,
    length_scale_from_affine,
    vessel_dynamics_fields,
)

_MM_TO_M = 0.001


def _twin(tmp_path, *, points_mm, edges, radii_mm, world_from_patient_m=None, omit=()):
    """Duck-typed PatientTwin exposing only what the adapter reads."""
    transform = np.eye(4) if world_from_patient_m is None else np.asarray(world_from_patient_m)
    artifacts = {}
    if "centerline_points" not in omit:
        artifacts["centerline_points"] = tmp_path / "points.npy"
        np.save(artifacts["centerline_points"], np.asarray(points_mm, dtype=np.float32))
    if "centerline_edges" not in omit:
        artifacts["centerline_edges"] = tmp_path / "edges.npy"
        np.save(artifacts["centerline_edges"], np.asarray(edges, dtype=np.int64))
    if "centerline_radii" not in omit and radii_mm is not None:
        artifacts["centerline_radii"] = tmp_path / "radii.npy"
        np.save(artifacts["centerline_radii"], np.asarray(radii_mm, dtype=np.float32))

    def patient_mm_to_world(points):
        points = np.asarray(points, dtype=np.float64)
        homogeneous = np.concatenate((points, np.ones((*points.shape[:-1], 1))), axis=-1)
        scaled = np.diag([_MM_TO_M, _MM_TO_M, _MM_TO_M, 1.0])
        return (homogeneous @ (scaled @ transform.T))[..., :3]

    return SimpleNamespace(
        artifacts=artifacts,
        world_from_patient_m=transform,
        patient_mm_to_world=patient_mm_to_world,
    )


def _straight_line(n: int = 5, spacing_mm: float = 10.0, radius_mm: float = 4.0):
    points = np.stack([np.arange(n) * spacing_mm, np.zeros(n), np.zeros(n)], axis=1).astype(np.float32)
    edges = np.stack([np.arange(n - 1), np.arange(1, n)], axis=1)
    radii = np.full(n, radius_mm, dtype=np.float32)
    return points, edges, radii


# --------------------------------------------------------------------------- #
# Length scale
# --------------------------------------------------------------------------- #
def test_identity_transform_has_unit_scale():
    assert length_scale_from_affine(np.eye(4)) == pytest.approx(1.0)


def test_rotation_alone_does_not_change_scale():
    angle = np.pi / 3.0
    transform = np.eye(4)
    transform[:3, :3] = [
        [np.cos(angle), -np.sin(angle), 0.0],
        [np.sin(angle), np.cos(angle), 0.0],
        [0.0, 0.0, 1.0],
    ]

    assert length_scale_from_affine(transform) == pytest.approx(1.0)


def test_uniform_scale_is_recovered():
    transform = np.diag([2.5, 2.5, 2.5, 1.0])

    assert length_scale_from_affine(transform) == pytest.approx(2.5)


def test_anisotropic_scale_is_rejected():
    """A scalar radius cannot describe an unevenly scaled lumen."""
    transform = np.diag([1.0, 2.0, 1.0, 1.0])

    with pytest.raises(ValueError, match="unequally"):
        length_scale_from_affine(transform)


# --------------------------------------------------------------------------- #
# Centerline conversion
# --------------------------------------------------------------------------- #
@pytest.fixture
def centerline_data_from_twin():
    pytest.importorskip(
        "catheter_vasculature_solver.vessel_deformation",
        reason="needs the catheter solver's vessel_deformation package",
    )
    from i4h_arena.medical.vessel_deformation import centerline_data_from_twin

    return centerline_data_from_twin


def test_converts_millimetres_to_metres(tmp_path, centerline_data_from_twin):
    points, edges, radii = _straight_line(spacing_mm=10.0, radius_mm=4.0)

    data = centerline_data_from_twin(_twin(tmp_path, points_mm=points, edges=edges, radii_mm=radii))

    # 10 mm spacing becomes 0.01 m, and a 4 mm radius becomes 0.004 m.
    np.testing.assert_allclose(data.starts[0], [0.0, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(data.ends[0], [0.01, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(data.start_radius_min[0], 0.004, atol=1e-9)


def test_radii_follow_the_transform_scale(tmp_path, centerline_data_from_twin):
    """Radii are lengths, so they must scale with the transform, not ignore it."""
    points, edges, radii = _straight_line(radius_mm=4.0)
    transform = np.diag([3.0, 3.0, 3.0, 1.0])

    data = centerline_data_from_twin(
        _twin(tmp_path, points_mm=points, edges=edges, radii_mm=radii, world_from_patient_m=transform)
    )

    np.testing.assert_allclose(data.start_radius_min[0], 0.004 * 3.0, atol=1e-9)


def test_translation_reaches_the_centerline(tmp_path, centerline_data_from_twin):
    points, edges, radii = _straight_line()
    transform = np.eye(4)
    transform[:3, 3] = [1.0, -2.0, 0.5]

    data = centerline_data_from_twin(
        _twin(tmp_path, points_mm=points, edges=edges, radii_mm=radii, world_from_patient_m=transform)
    )

    np.testing.assert_allclose(data.starts[0], [1.0, -2.0, 0.5], atol=1e-9)


def test_edges_select_their_endpoints(tmp_path, centerline_data_from_twin):
    points, _, radii = _straight_line(n=4)
    # Deliberately out of order, to catch an implementation that assumes
    # consecutive indexing instead of honouring the edge list.
    edges = np.array([[2, 3], [0, 1]], dtype=np.int64)

    data = centerline_data_from_twin(_twin(tmp_path, points_mm=points, edges=edges, radii_mm=radii))

    np.testing.assert_allclose(data.starts[0], [0.02, 0.0, 0.0], atol=1e-9)
    np.testing.assert_allclose(data.starts[1], [0.0, 0.0, 0.0], atol=1e-9)


def test_missing_centerline_yields_no_vessel(tmp_path, centerline_data_from_twin):
    """Phantom scenes have no centerline and must still run."""
    points, edges, radii = _straight_line()

    twin = _twin(tmp_path, points_mm=points, edges=edges, radii_mm=radii, omit=("centerline_points",))

    assert centerline_data_from_twin(twin) is None


def test_missing_radii_is_an_error_not_a_silent_skip(tmp_path, centerline_data_from_twin):
    """Without radii there is no wall, so failing loudly beats a vessel-free run."""
    points, edges, _ = _straight_line()

    twin = _twin(tmp_path, points_mm=points, edges=edges, radii_mm=None)

    with pytest.raises(ValueError, match="centerline_radii"):
        centerline_data_from_twin(twin)


def test_radii_count_must_match_points(tmp_path, centerline_data_from_twin):
    points, edges, _ = _straight_line(n=5)

    twin = _twin(tmp_path, points_mm=points, edges=edges, radii_mm=np.full(3, 4.0))

    with pytest.raises(ValueError, match="one radius per node"):
        centerline_data_from_twin(twin)


def test_out_of_range_edge_is_rejected(tmp_path, centerline_data_from_twin):
    points, _, radii = _straight_line(n=4)

    twin = _twin(tmp_path, points_mm=points, edges=np.array([[0, 9]], dtype=np.int64), radii_mm=radii)

    with pytest.raises(ValueError, match="outside centerline_points"):
        centerline_data_from_twin(twin)


def test_nonpositive_radius_is_rejected(tmp_path, centerline_data_from_twin):
    points, edges, radii = _straight_line(n=4)
    radii[1] = 0.0

    twin = _twin(tmp_path, points_mm=points, edges=edges, radii_mm=radii)

    with pytest.raises(ValueError, match="strictly positive"):
        centerline_data_from_twin(twin)


# --------------------------------------------------------------------------- #
# Wall dynamics
#
# The boundary conditions and damping decide whether the wall stays on the
# anatomy, and none of it is visible from the rod's own state: the catheter can
# read as perfectly shaped while the lumen it sits in has been carried away.
# --------------------------------------------------------------------------- #
def test_both_ends_are_anchored_by_default():
    """With only the root held, a wall whose bend stiffness is 1.0 swings about
    that anchor instead of bending, which reads as the wire leaving the lumen."""
    fields = vessel_dynamics_fields()

    assert fields["root_locked"] is True
    assert fields["endpoints_locked"] is True


def test_the_wall_is_damped_by_default():
    fields = vessel_dynamics_fields()

    assert fields["linear_damping"] == pytest.approx(0.01)
    assert fields["angular_damping"] == pytest.approx(0.01)


def test_the_root_stays_anchored_even_with_the_distal_end_free():
    """Freeing the distal end is a legitimate comparison; losing the root is
    not, since then nothing registers the vessel to the anatomy at all."""
    fields = vessel_dynamics_fields(endpoints_locked=False)

    assert fields["endpoints_locked"] is False
    assert fields["root_locked"] is True


@pytest.mark.parametrize("name", ["linear_damping", "angular_damping"])
@pytest.mark.parametrize("value", [-0.01, 1.5])
def test_damping_outside_the_unit_range_is_rejected(name, value):
    with pytest.raises(ValueError, match=name):
        vessel_dynamics_fields(**{name: value})


def test_the_reference_values_are_what_the_helper_defaults_to():
    """The constants are what callers and the spec cite, so a change to one that
    misses the other would leave the two disagreeing about the same wall."""
    assert VESSEL_ENDPOINTS_LOCKED is True
    assert VESSEL_RESPONSE == pytest.approx(0.5)
    assert VESSEL_LINEAR_DAMPING == pytest.approx(0.01)
    assert VESSEL_ANGULAR_DAMPING == pytest.approx(0.01)


def test_every_field_the_helper_names_is_a_real_solver_parameter():
    """The helper spells these as strings, so a renamed solver field would
    otherwise surface as a TypeError only once a GPU scene tried to build."""
    pytest.importorskip(
        "catheter_vasculature_solver.vessel_deformation",
        reason="needs the catheter solver's vessel_deformation package",
    )
    import dataclasses

    from catheter_vasculature_solver.vessel_deformation import CenterlineDynamicsParams

    known = {field.name for field in dataclasses.fields(CenterlineDynamicsParams)}

    assert set(vessel_dynamics_fields()) <= known


def test_the_wall_takes_half_of_a_contact_correction_by_default():
    """At 1.0 the wall yields completely and the wire is contained by nothing."""
    import inspect

    from i4h_arena.medical.vessel_deformation import centerline_vessel_from_twin

    defaults = inspect.signature(centerline_vessel_from_twin).parameters

    assert defaults["vessel_response"].default == pytest.approx(0.5)
    assert defaults["two_way"].default is True

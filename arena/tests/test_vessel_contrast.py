# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The patient-twin side of vessel contrast, and the renderer that composites it.

The carve, the transport and the chord integrator are tested in xray_simulator, which owns them.
"""

from __future__ import annotations

import numpy as np
import pytest
from xray_simulator import HuToMuMapping
from xray_simulator.geometry import project_point_to_detector
from xray_simulator.vessel import ContrastAgent, HuToMu, VesselState

from i4h_arena.medical.carm import CArmState
from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import DEFAULT_HU_TO_MU_PRESET, PatientVolume
from i4h_arena.medical.slang_fluoroscopy import SlangFluoroscopyRenderer, solve_projection_geometry
from i4h_arena.medical.vessel_motion import VesselMotion
from i4h_arena.medical.vessel_twin import hu_to_mu, ijk_to_volume_mm, vessel_tree, volume_scale

SHAPE_IJK = (24, 24, 24)
SPACING_MM = 1.0
#: A scan origin away from zero, so a carve that assumed voxel ``i`` sits at ``i * spacing`` lands
#: off the centerline.
ORIGIN_MM = (-11.0, 7.0, 3.0)
LUMEN_HU = 375.0
TISSUE_HU = 120.0
_CURVE = HuToMu.from_mapping(HuToMuMapping.preset(DEFAULT_HU_TO_MU_PRESET))
LUMEN_MU = float(_CURVE.mu_of(LUMEN_HU))
TISSUE_MU = float(_CURVE.mu_of(TISSUE_HU))


def _voxel_to_lps_mm() -> np.ndarray:
    affine = np.diag((SPACING_MM, SPACING_MM, SPACING_MM, 1.0))
    affine[:3, 3] = ORIGIN_MM
    return affine


@pytest.fixture
def tube_patient(write_patient_bundle):
    """Build a straight contrast-filled tube along i through a block of soft tissue."""

    def build(
        *, radius_mm: float = 3.0, with_mask: bool = True, with_centerline: bool = True, preset: str | None = None
    ) -> PatientVolume:
        i, j, k = np.indices(SHAPE_IJK)
        centre = (SHAPE_IJK[0] - 1) / 2.0
        inside = ((j - centre) ** 2 + (k - centre) ** 2) <= radius_mm**2
        hu = np.full(SHAPE_IJK, TISSUE_HU, dtype=np.float32)
        hu[inside] = LUMEN_HU
        nodes = SHAPE_IJK[0]
        index = np.zeros((nodes, 4))
        index[:, 0] = np.arange(nodes)
        index[:, 1:3] = centre
        index[:, 3] = 1.0
        points = (index @ _voxel_to_lps_mm().T)[:, :3].astype(np.float32)
        centerline = (
            points,
            np.stack([np.arange(nodes - 1), np.arange(1, nodes)], axis=-1),
            np.full(nodes, radius_mm, dtype=np.float32),
        )
        manifest = write_patient_bundle(
            hu,
            _voxel_to_lps_mm(),
            name="tube",
            vessel_mask_ijk=inside if with_mask else None,
            centerline_scan=centerline if with_centerline else None,
            world_from_scan_m=np.eye(4),
        )
        return PatientVolume.load(PatientTwin.load(manifest), hu_to_mu_preset=preset)

    return build


# ---------------------------------------------------------------- twin adapter


def test_volume_scale_is_measured_through_the_public_transform_chain(tube_patient) -> None:
    # A millimetre of patient is a millimetre of volume here, and would be a thousand times
    # larger if the metre-scaled placement were composed with the millimetre-scaled one.
    assert volume_scale(tube_patient()) == pytest.approx(1.0)


def test_the_tree_is_read_in_volume_millimetres_and_smoothed(tube_patient) -> None:
    patient = tube_patient()
    tree = vessel_tree(patient)
    stored = np.load(patient.twin.artifacts["centerline_points"])
    # A straight run only moves at its ends under smoothing; the interior stays on the axis.
    np.testing.assert_allclose(tree.points_mm[2:-2], stored[2:-2], atol=1e-5)
    np.testing.assert_allclose(tree.radii_mm, 3.0)


def test_the_carve_grid_is_the_scan_affine(tube_patient) -> None:
    patient = tube_patient()
    np.testing.assert_allclose(ijk_to_volume_mm(patient), _voxel_to_lps_mm())


def test_the_tree_requires_one_radius_per_node(tube_patient) -> None:
    patient = tube_patient()
    np.save(patient.twin.artifacts["centerline_radii"], np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="one radius per node"):
        vessel_tree(patient)


@pytest.mark.parametrize("preset", [None, "linear"])
def test_the_curve_is_the_preset_the_volume_was_built_with(tube_patient, preset) -> None:
    patient = tube_patient(preset=preset)
    expected = HuToMu.from_mapping(HuToMuMapping.preset(preset or DEFAULT_HU_TO_MU_PRESET))
    curve = hu_to_mu(patient)
    np.testing.assert_array_equal(curve.hu, expected.hu)
    np.testing.assert_array_equal(curve.mu, expected.mu)


# ---------------------------------------------------------------- renderer


def _carm(patient: PatientVolume) -> CArmState:
    centre = patient.volume_mm_to_world(patient.center_xyz_mm)
    return CArmState(
        source_world_m=np.asarray([centre + [0.0, 0.0, -0.10]]),
        detector_center_world_m=np.asarray([centre + [0.0, 0.0, 0.10]]),
        detector_x_axis_world=np.asarray([[1.0, 0.0, 0.0]]),
        detector_size_m=(0.064, 0.064),
    )


def _renderer(patient: PatientVolume, **kwargs) -> SlangFluoroscopyRenderer:
    return SlangFluoroscopyRenderer(
        patient,
        _carm(patient),
        width=64,
        height=64,
        step_mm=1.0,
        device_type="vulkan",
        catheter_attenuation=False,
        **kwargs,
    )


def _catheter(patient: PatientVolume) -> CatheterState:
    centre = patient.volume_mm_to_world(patient.center_xyz_mm)
    return CatheterState(
        positions_world_m=np.asarray([[centre + [-0.005, 0.0, 0.0], centre + [0.005, 0.0, 0.0]]]),
        valid_nodes=np.asarray([2]),
    )


def test_composited_geometry_uses_the_rays_the_volume_is_marched_along(tube_patient) -> None:
    """Rays built from the projection's axes end on the pixel the shader's pose projects onto.

    The renderer is handed Euler angles and a translation; absorbers are handed the axes those
    were solved from. Disagreeing by a pixel would register every composited vessel and
    catheter a pixel off the anatomy.
    """
    patient = tube_patient()
    carm = CArmState(
        source_world_m=np.asarray([[0.31, -0.42, 0.05]]),
        detector_center_world_m=np.asarray([[-0.30, 0.25, 0.11]]),
        detector_x_axis_world=np.asarray([[0.2, 0.3, 0.9]]),
        detector_size_m=(0.064, 0.048),
    )
    projection = solve_projection_geometry(patient, carm, width=64, height=48)
    rays = projection.rays(width=64, height=48)
    centers = rays.pixel_centers_mm()
    for row, column in ((0, 0), (20, 50), (47, 63)):
        point = rays.source_mm + 0.5 * (centers[row, column] - rays.source_mm)
        projected = project_point_to_detector(
            point,
            projection.rotation_zxy_rad,
            projection.translation_xyz_mm,
            projection.source_to_detector_mm,
            projection.source_to_isocenter_mm,
            64,
            48,
            projection.pixel_spacing_mm,
            patient.center_xyz_mm,
        )
        assert projected == pytest.approx((column, row), abs=1e-4)


def test_the_carve_leaves_the_rendered_volume_without_its_lumen(tube_patient) -> None:
    patient = tube_patient()
    vessel = _renderer(patient, vessel_contrast=True)._vessel
    assert vessel is not None
    assert vessel.carve.carved_voxels > 0
    assert vessel.carve.contrast_mu_per_mm == pytest.approx(LUMEN_MU - TISSUE_MU, rel=0.15)
    np.testing.assert_allclose(vessel.rest_state.mu_per_mm, LUMEN_MU - TISSUE_MU, rtol=0.15)


def test_rendering_without_a_vessel_state_falls_back_to_the_scan_itself(tube_patient) -> None:
    patient = tube_patient()
    frames = _renderer(patient, vessel_contrast=True).render(_catheter(patient))
    for key in ("rgb", "guidance", "dsa", "dsa_guidance"):
        assert frames[key].shape == (1, 64, 64, 3)
    assert not bool(frames["attenuation"].isnan().any())


def test_the_subtracted_channel_keeps_its_fixed_window(tube_patient) -> None:
    """The window decided from the reference injection must survive construction.

    Fitting it to the first frame instead would land it on an empty, pre-injection image.
    """
    patient = tube_patient()
    renderer = _renderer(patient, vessel_contrast=True, dsa=True, dsa_reference_mg_per_ml=4.0)
    assert renderer._dsa_display is not None
    assert renderer._dsa_display.scaling == "log"
    assert renderer._dsa_display.log_window[1] > 0.0


def test_moving_the_vessel_shows_up_far_more_in_the_subtracted_channel(tube_patient) -> None:
    """The reason DSA is a subtraction and not a brightness boost.

    The carve leaves a static residual, because a centerline tube cannot describe a real lumen
    exactly. That residual is in the mask and in the live frame identically, so it divides out of
    the subtraction and survives in the plain view.
    """
    patient = tube_patient()
    # Saturate the subtracted channel at the tube's own iodine rather than a clinical injection,
    # which on the interventional curve would leave this small phantom near the bottom of the window.
    tube_mg_per_ml = ContrastAgent().concentration_matching(LUMEN_MU - TISSUE_MU, baseline_mu=TISSUE_MU, curve=_CURVE)
    renderer = _renderer(patient, vessel_contrast=True, dsa=True, dsa_reference_mg_per_ml=tube_mg_per_ml)
    rest = renderer._vessel.rest_state
    moved = VesselState(
        rest.points_mm + np.array([0.0, 1.5, 0.0], dtype=np.float32),
        rest.edges,
        rest.radii_mm,
        rest.mu_per_mm,
    )
    before = renderer.render(_catheter(patient))
    after = renderer.render(_catheter(patient), vessel=moved)
    plain = (after["rgb"].float() - before["rgb"].float()).abs().mean()
    subtracted = (after["dsa"].float() - before["dsa"].float()).abs().mean()
    assert subtracted > plain


def test_solver_motion_arrives_in_volume_millimetres(tube_patient) -> None:
    patient = tube_patient()
    renderer = _renderer(patient, vessel_contrast=True)
    rest = renderer._vessel.rest_state
    displacement = np.zeros((1, rest.points_mm.shape[1], 3))
    displacement[..., 1] = 0.002
    state = renderer.vessel_state(VesselMotion(displacement_world_m=displacement))
    np.testing.assert_allclose(
        state.points_mm - rest.points_mm, np.broadcast_to([0.0, 2.0, 0.0], state.points_mm.shape), atol=1e-4
    )


def test_contrast_rendering_needs_a_centerline(tube_patient) -> None:
    patient = tube_patient(with_centerline=False)
    with pytest.raises(ValueError, match="centerline"):
        _renderer(patient, vessel_contrast=True)


def test_a_vessel_state_for_the_wrong_batch_is_refused(tube_patient) -> None:
    patient = tube_patient()
    renderer = _renderer(patient, vessel_contrast=True)
    rest = renderer._vessel.rest_state
    three = VesselState(
        np.repeat(rest.points_mm, 3, axis=0),
        rest.edges,
        np.repeat(rest.radii_mm, 3, axis=0),
        np.repeat(rest.mu_per_mm, 3, axis=0),
    )
    with pytest.raises(ValueError, match="covers 3 environments"):
        renderer.render(_catheter(patient), vessel=three)

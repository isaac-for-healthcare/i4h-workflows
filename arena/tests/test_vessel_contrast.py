# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json

import numpy as np
import pytest
import yaml

from i4h_arena.medical.carm import CArmState
from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.contrast_bolus import (
    ContrastAgent,
    ContrastTransport,
    HuToMu,
    Injection,
)
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.medical.slang_fluoroscopy import ProjectionGeometry, SlangFluoroscopyRenderer
from i4h_arena.medical.vessel_attenuation import VesselAttenuation, VesselState
from i4h_arena.medical.vessel_carve import (
    carve_lumen,
    carve_residual,
    centerline_in_volume_mm,
    node_contrast_mu_per_mm,
    smooth_centerline,
    tube_occupancy,
    volume_scale,
)

# A beam down +z with the isocenter at the origin, so a chord can be read off a column by hand.
DETECTOR_PX = 64
PIXEL_MM = 1.0
SID_MM = 500.0
SDD_MM = 1000.0
MAGNIFICATION = SDD_MM / SID_MM
CENTER_COLUMN = 0.5 * DETECTOR_PX - 0.5
CENTER_ROW = DETECTOR_PX // 2

SHAPE_ZYX = (24, 24, 24)
SPACING_MM = 1.0
LUMEN_MU = 0.004
TISSUE_MU = 0.001


def _projection() -> ProjectionGeometry:
    return ProjectionGeometry(
        rotation_zxy_rad=(0.0, 0.0, 0.0),
        translation_xyz_mm=(0.0, 0.0, 0.0),
        source_to_detector_mm=SDD_MM,
        source_to_isocenter_mm=SID_MM,
        pixel_spacing_mm=PIXEL_MM,
        local_to_volume=np.eye(3),
        isocenter_volume_mm=np.zeros(3),
    )


def _column_of(volume_x_mm: float) -> int:
    return int(round(MAGNIFICATION * volume_x_mm + CENTER_COLUMN))


# The detector has an even pixel count, so no pixel centre sits exactly on the axis: the nearest
# row and column are half a pixel off, which at this magnification is a quarter of a millimetre
# in the volume. Small, but a ray that misses the axis by 0.25 mm cuts a shorter chord, and
# expecting the full diameter would read as a kernel error on a narrow vessel.
_AXIS_OFFSET_MM = 0.5 * PIXEL_MM / MAGNIFICATION


def _chord_through(radius_mm: float) -> float:
    """Chord a ray offset from the axis by half a pixel cuts through a cylinder."""
    return 2.0 * np.sqrt(radius_mm**2 - _AXIS_OFFSET_MM**2)


def _tube_patient(
    tmp_path,
    *,
    radius_mm: float = 3.0,
    with_mask: bool = True,
    with_centerline: bool = True,
) -> PatientVolume:
    """A straight contrast-filled tube along x through a block of soft tissue.

    Synthetic so the carve's answers are known: the lumen is exactly a cylinder of one
    attenuation, so what has to come out and what has to go back are both analytic.
    """
    z, y, x = np.indices(SHAPE_ZYX)
    centre = (SHAPE_ZYX[0] - 1) / 2.0
    inside = ((y - centre) ** 2 + (z - centre) ** 2) <= radius_mm**2
    mu = np.full(SHAPE_ZYX, TISSUE_MU, dtype=np.float32)
    mu[inside] = LUMEN_MU
    np.save(tmp_path / "mu_volume.npy", mu)
    (tmp_path / "metadata.json").write_text(
        json.dumps(
            {
                "shape_zyx": list(SHAPE_ZYX),
                "spacing_zyx_mm": [SPACING_MM] * 3,
                "hu_to_mu": {
                    "control_points": [[-1000.0, 0.0], [-300.0, 0.0], [100.0, 0.0008], [300.0, 0.0028]],
                },
            }
        ),
        encoding="utf-8",
    )
    # One node per voxel. The tube is a sweep of per-node spheres, so sparse nodes scallop the
    # surface inward by about h^2 / 8r between them; at this spacing that is well under a tenth
    # of a voxel, which is the regime real centerlines sit in.
    nodes = SHAPE_ZYX[2]
    points = np.zeros((nodes, 3), dtype=np.float32)
    # Spanning the full extent, so the tube covers the whole segmented cylinder and coverage is
    # a property of the radii rather than of where the centerline happens to stop.
    points[:, 0] = np.linspace(0.0, (SHAPE_ZYX[2] - 1) * SPACING_MM, nodes)
    points[:, 1] = centre * SPACING_MM
    points[:, 2] = centre * SPACING_MM
    np.save(tmp_path / "centerline_points_mm.npy", points)
    np.save(tmp_path / "centerline_radii_mm.npy", np.full(nodes, radius_mm, dtype=np.float32))
    np.save(tmp_path / "centerline_edges.npy", np.stack([np.arange(nodes - 1), np.arange(1, nodes)], axis=-1))

    artifacts = {"attenuation_volume": "mu_volume.npy", "volume_metadata": "metadata.json"}
    if with_centerline:
        artifacts |= {
            "centerline_points": "centerline_points_mm.npy",
            "centerline_radii": "centerline_radii_mm.npy",
            "centerline_edges": "centerline_edges.npy",
        }
    if with_mask:
        np.save(tmp_path / "vessel_mask.npy", inside.astype(np.uint8))
        artifacts["vessel_mask"] = "vessel_mask.npy"
    (tmp_path / "patient_twin.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "patient_id": "tube",
                "coordinate_frame": "DICOM_LPS",
                "transforms": {
                    "voxel_to_patient_mm": np.diag((SPACING_MM, SPACING_MM, SPACING_MM, 1.0)).tolist(),
                    "world_from_patient_m": np.eye(4).tolist(),
                },
                "artifacts": artifacts,
            }
        ),
        encoding="utf-8",
    )
    return PatientVolume.load(PatientTwin.load(tmp_path / "patient_twin.yaml"))


# ---------------------------------------------------------------- carve


def test_volume_scale_is_measured_through_the_public_transform_chain(tmp_path) -> None:
    # A millimetre of patient is a millimetre of volume here, and would be a thousand times
    # larger if the metre-scaled placement were composed with the millimetre-scaled one.
    assert volume_scale(_tube_patient(tmp_path)) == pytest.approx(1.0)


def test_tube_occupancy_matches_the_cylinder_it_describes(tmp_path) -> None:
    patient = _tube_patient(tmp_path, radius_mm=3.0)
    occupancy = tube_occupancy(patient)
    # Away from the capped ends the cross-section is a disc of the node radius.
    column = occupancy[:, :, SHAPE_ZYX[2] // 2]
    assert column.sum() == pytest.approx(np.pi * 9.0, rel=0.15)


def test_supersampling_resolves_the_boundary_fractionally(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    coarse = tube_occupancy(patient, supersample=1)
    fine = tube_occupancy(patient, supersample=3)
    assert set(np.unique(coarse)) <= {0.0, 1.0}
    assert np.any((fine > 0.0) & (fine < 1.0)), "supersampling should produce partial voxels"
    assert fine.sum() == pytest.approx(coarse.sum(), rel=0.2)


def test_carve_removes_the_lumen_and_leaves_everything_else(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    original = patient.mu_volume.copy()
    result = carve_lumen(patient)
    carved = result.occupancy > 0.0

    assert result.lumen_mu_per_mm == pytest.approx(LUMEN_MU, rel=0.05)
    assert result.fill_mu_per_mm == pytest.approx(TISSUE_MU, rel=0.1)
    assert result.contrast_mu_per_mm == pytest.approx(LUMEN_MU - TISSUE_MU, rel=0.1)
    assert np.array_equal(result.mu_volume[~carved], original[~carved])
    assert patient.mu_volume is not result.mu_volume
    np.testing.assert_array_equal(patient.mu_volume, original)


def test_carve_reports_how_much_of_the_segmentation_the_tube_covers(tmp_path) -> None:
    patient = _tube_patient(tmp_path, with_mask=True)
    # Reported even when the union is not carved, because it says how much contrast the
    # centerline model cannot represent at all.
    assert carve_lumen(patient, include_segmentation=False).tube_covers_mask == pytest.approx(1.0, abs=0.1)
    assert carve_lumen(patient, include_segmentation=True).tube_covers_mask == pytest.approx(1.0, abs=0.1)


def test_carve_without_a_mask_reports_no_coverage(tmp_path) -> None:
    assert carve_lumen(_tube_patient(tmp_path, with_mask=False)).tube_covers_mask is None


def test_carving_the_union_removes_more_than_the_tube_can_restore(tmp_path) -> None:
    patient = _tube_patient(tmp_path, with_mask=True)
    tube_only = carve_lumen(patient, include_segmentation=False)
    union = carve_lumen(patient, include_segmentation=True)
    assert union.carved_voxels >= tube_only.carved_voxels


def test_restoring_the_carve_reproduces_the_volume_it_came_from(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    result = carve_lumen(patient)
    residual = carve_residual(patient, result)
    # The synthetic lumen is uniform, so putting it back is near-exact; the bar is that the
    # residual sits well under the effect it would otherwise swamp.
    assert residual["rms"] < 0.25 * residual["signal"]
    assert residual["max"] < 2.0 * residual["signal"]


def test_node_coefficients_recover_the_lumen_the_carve_removed(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    result = carve_lumen(patient)
    per_node = node_contrast_mu_per_mm(patient, result)
    assigned = per_node[per_node > 0.0]
    assert len(assigned) > 0
    assert assigned.mean() == pytest.approx(LUMEN_MU - TISSUE_MU, rel=0.15)


def test_carve_rejects_a_mask_that_does_not_match_the_volume(tmp_path) -> None:
    patient = _tube_patient(tmp_path, with_mask=True)
    np.save(tmp_path / "vessel_mask.npy", np.zeros((3, 3, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="does not match"):
        carve_lumen(patient, include_segmentation=True)


def test_carve_rejects_nonsense_donor_quantiles(tmp_path) -> None:
    with pytest.raises(ValueError, match="donor_quantiles"):
        carve_lumen(_tube_patient(tmp_path), donor_quantiles=(0.8, 0.2))


def test_centerline_requires_one_radius_per_node(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    np.save(tmp_path / "centerline_radii_mm.npy", np.ones(3, dtype=np.float32))
    with pytest.raises(ValueError, match="one radius per node"):
        centerline_in_volume_mm(patient)


# ---------------------------------------------------------------- transport


def _curve() -> HuToMu:
    return HuToMu.from_metadata(
        {"hu_to_mu": {"control_points": [[-1000.0, 0.0], [-300.0, 0.0], [100.0, 0.0008], [300.0, 0.0028]]}}
    )


def test_transfer_function_inverts_on_its_strictly_increasing_part() -> None:
    curve = _curve()
    for hu in (100.0, 200.0, 300.0):
        assert float(curve.hu_of(curve.mu_of(hu))) == pytest.approx(hu)


def test_inverting_a_flat_run_keeps_its_upper_end() -> None:
    # Zero attenuation spans -1000..-300 HU. Collapsing the run to its first point instead would
    # stretch the next segment back to -1000 and drag soft tissue hundreds of HU negative.
    curve = _curve()
    assert float(curve.hu_of(0.0)) == pytest.approx(-300.0)
    assert float(curve.hu_of(0.0004)) == pytest.approx(-100.0, abs=1.0)


def test_concentration_is_fitted_from_the_attenuation_it_has_to_reproduce() -> None:
    curve, agent = _curve(), ContrastAgent()
    baseline = 0.001
    concentration = agent.concentration_matching(0.0005, baseline_mu=baseline, curve=curve)
    replayed = agent.mu_excess(np.array([concentration]), baseline_mu=baseline, curve=curve)[0]
    assert replayed == pytest.approx(0.0005, abs=1e-9)


def _tree() -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """A trunk that bifurcates: 10 nodes down, then two branches of 5."""
    trunk = np.stack([np.zeros(10), np.zeros(10), np.arange(10.0)], axis=-1)
    left = np.stack([np.arange(1.0, 6.0), np.zeros(5), np.full(5, 9.0)], axis=-1)
    right = np.stack([-np.arange(1.0, 6.0), np.zeros(5), np.full(5, 9.0)], axis=-1)
    points = np.concatenate([trunk, left, right])
    edges = [(index, index + 1) for index in range(9)]
    edges += [(9, 10)] + [(10 + index, 11 + index) for index in range(4)]
    edges += [(9, 15)] + [(15 + index, 16 + index) for index in range(4)]
    radii = np.concatenate([np.full(10, 4.0), np.full(5, 3.0), np.full(5, 2.0)])
    return points, np.asarray(edges), radii


def test_flow_splits_by_murrays_law_at_a_bifurcation() -> None:
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    flow = transport.flow_ml_per_s
    assert flow[0] == pytest.approx(10.0)
    assert flow[10] + flow[15] == pytest.approx(10.0)
    assert flow[10] / flow[15] == pytest.approx(3.0**3 / 2.0**3)


def test_flow_is_conserved_along_an_unbranched_run() -> None:
    points, edges, radii = _tree()
    flow = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0).flow_ml_per_s
    np.testing.assert_allclose(flow[:10], 10.0)


def test_contrast_fills_downstream_and_then_clears() -> None:
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    injection = Injection(node=0, start_s=0.0, duration_s=0.2, flow_ml_per_s=10.0)
    peak = 0.0
    for _ in range(40):
        transport.step(0.05, injection)
        peak = max(peak, float(transport.concentration_mg_per_ml.max()))
    assert peak > 1.0, "the injection should raise concentration somewhere"
    assert float(transport.concentration_mg_per_ml.max()) < 0.01 * peak, "and it should wash out"


def test_the_root_is_flushed_by_contrast_free_blood() -> None:
    # Falling back to the root's own concentration makes it a perfect reservoir with nothing to
    # flush it, and the bolus never clears.
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    transport.step(0.1, Injection(node=0, start_s=0.0, duration_s=0.1, flow_ml_per_s=10.0))
    loaded = float(transport.concentration_mg_per_ml[0])
    transport.step(2.0, None)
    assert float(transport.concentration_mg_per_ml[0]) < 0.01 * loaded


def test_advection_carries_contrast_downstream_only() -> None:
    # With dispersion off the transport is pure advection, which is one-directional: contrast
    # delivered mid-tree fills what is below the tip and cannot reach the root or the sibling
    # branch. Rooting the tree at the injection site instead would send it everywhere.
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0, dispersion_mm2_per_s=0.0)
    transport.step(0.4, Injection(node=12, start_s=0.0, duration_s=0.4, flow_ml_per_s=10.0))
    concentration = transport.concentration_mg_per_ml
    assert concentration[13] > 1.0, "the branch below the tip should fill"
    assert concentration[:10].max() == pytest.approx(0.0), "the trunk above it must not"
    assert concentration[15:].max() == pytest.approx(0.0), "nor the sibling branch"


def test_dispersion_spreads_both_ways_but_advection_dominates() -> None:
    # Dispersion is diffusive, so a little contrast does work upstream; that is the model being
    # honest rather than a leak. What must hold is that it stays small next to the advected front.
    points, edges, radii = _tree()
    transport = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    transport.step(0.4, Injection(node=12, start_s=0.0, duration_s=0.4, flow_ml_per_s=10.0))
    concentration = transport.concentration_mg_per_ml
    assert concentration[:10].max() > 0.0
    assert concentration[:10].max() < 0.25 * concentration[13]


def test_inflow_defaults_to_the_widest_endpoint() -> None:
    points, edges, radii = _tree()
    assert ContrastTransport(points, edges, radii).inflow_node == 0


def test_transport_rejects_a_disconnected_centerline() -> None:
    points, edges, radii = _tree()
    with pytest.raises(ValueError, match="disconnected"):
        ContrastTransport(points, edges[:3], radii, inflow=0)


def test_injection_concentration_is_diluted_by_the_blood_it_enters() -> None:
    points, edges, radii = _tree()
    agent = ContrastAgent(stock_mg_per_ml=300.0)
    fast = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    fast.step(0.3, Injection(node=0, start_s=0.0, duration_s=0.3, flow_ml_per_s=10.0), agent=agent)
    slow = ContrastTransport(points, edges, radii, inflow=0, flow_ml_per_s=10.0)
    slow.step(0.3, Injection(node=0, start_s=0.0, duration_s=0.3, flow_ml_per_s=1.0), agent=agent)
    assert float(fast.concentration_mg_per_ml.max()) == pytest.approx(300.0 * 10.0 / 20.0, rel=0.05)
    assert float(slow.concentration_mg_per_ml.max()) < 0.3 * float(fast.concentration_mg_per_ml.max())


# ---------------------------------------------------------------- chord kernel


def _chord_image(radii: tuple[float, float], mus: tuple[float, float] = (1.0, 1.0)) -> np.ndarray:
    points = np.array([[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=np.float32)
    state = VesselState(points, np.array([[0, 1]]), np.asarray(radii), np.asarray(mus))
    return VesselAttenuation(device="cpu").line_integral(
        state, _projection(), width=DETECTOR_PX, height=DETECTOR_PX
    )


def test_a_ray_across_a_cylinder_sees_its_diameter() -> None:
    image = _chord_image((4.0, 4.0))
    assert image[0, CENTER_ROW, _column_of(0.0)] == pytest.approx(_chord_through(4.0), abs=0.02)


def test_a_ray_past_the_end_planes_sees_nothing() -> None:
    # The segment spans x in [-10, 10] and the detector reaches about 15 mm of volume x.
    image = _chord_image((4.0, 4.0))
    assert image[0, CENTER_ROW, _column_of(13.0)] == pytest.approx(0.0)


def test_the_chord_follows_a_taper_along_the_segment() -> None:
    # Radius runs 2 -> 10 over x in [-10, 10], so it is 6 + 0.4x. The sampled column is half a
    # pixel off the axis, so the radius is read where the ray actually crosses.
    image = _chord_image((2.0, 10.0))
    for volume_x in (-5.0, 0.0, 5.0):
        axial_x = (_column_of(volume_x) - CENTER_COLUMN) / MAGNIFICATION
        expected = _chord_through(6.0 + 0.4 * axial_x)
        assert image[0, CENTER_ROW, _column_of(volume_x)] == pytest.approx(expected, abs=0.05)


def test_a_bent_chain_is_not_counted_twice_at_its_joints() -> None:
    """Two segments meeting at an angle must sum to the chord through their union.

    Clipped to planes perpendicular to each segment's own axis, the two shapes overlap on the
    inside of the turn and leave a wedge on the outside, and summing their chords counts the
    overlap twice. On a real centerline that alternates along the vessel and renders as a
    barcode, so the chords are checked here against a marched union rather than each other.
    """
    radius = 4.0
    points = np.array([[[-10.0, 0.0, 0.0], [0.0, 0.0, 0.0], [8.0, 6.0, 0.0]]], dtype=np.float32)
    state = VesselState(points, np.array([[0, 1], [1, 2]]), np.full(3, radius), np.ones(3))
    image = VesselAttenuation(device="cpu").line_integral(
        state, _projection(), width=DETECTOR_PX, height=DETECTOR_PX
    )[0]

    # Sampled around the joint and out towards the wall, because the discrepancy is zero on the
    # axis and grows to r*tan(turn / 2) at the lumen's edge.
    nodes = points[0].astype(np.float64)
    step = 0.02
    depth = np.arange(-30.0, 30.0, step)
    for volume_x in (-3.0, 0.0, 3.0):
        for volume_y in (-3.0, 0.0, 3.0):
            column, row = _column_of(volume_x), _column_of(volume_y)
            position = (np.array([column, row]) - CENTER_COLUMN) / MAGNIFICATION
            along = np.stack([np.full_like(depth, position[0]), np.full_like(depth, position[1]), depth], axis=-1)
            inside = np.zeros(len(depth), dtype=bool)
            for first, second in ((nodes[0], nodes[1]), (nodes[1], nodes[2])):
                axis = second - first
                fraction = np.clip(((along - first) @ axis) / (axis @ axis), 0.0, 1.0)
                nearest = first + fraction[:, None] * axis
                inside |= np.linalg.norm(along - nearest, axis=-1) <= radius
            assert image[row, column] == pytest.approx(inside.sum() * step, abs=0.1)


def test_smoothing_straightens_the_voxel_staircase_without_moving_the_vessel() -> None:
    """A node is pulled off the staircase by less than the voxel the staircase came from."""
    staircase = np.array([[float(index // 2), float(index % 2), 0.0] for index in range(12)])
    edges = np.stack([np.arange(11), np.arange(1, 12)], axis=-1)
    smoothed = smooth_centerline(staircase, edges)

    def mean_turn(points: np.ndarray) -> float:
        before = points[1:-1] - points[:-2]
        after = points[2:] - points[1:-1]
        cosine = np.sum(before * after, axis=-1) / (
            np.linalg.norm(before, axis=-1) * np.linalg.norm(after, axis=-1)
        )
        return float(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))).mean())

    assert mean_turn(smoothed) < 0.5 * mean_turn(staircase)
    assert np.linalg.norm(smoothed - staircase, axis=-1).max() < 1.0


def test_a_taper_is_not_silently_treated_as_a_cylinder() -> None:
    tapered = _chord_image((2.0, 10.0))
    straight = _chord_image((6.0, 6.0))
    assert np.abs(tapered - straight).max() > 1.0


def test_a_ray_inside_the_cones_own_half_angle_is_still_contained() -> None:
    """The branch where the quadratic's leading coefficient goes negative.

    A steeply tapered segment pointing along the beam puts the ray inside the cone's half-angle,
    where containment is the region outside the roots rather than between them. Clamping that
    away would silently drop the chord.
    """
    points = np.array([[[0.0, 0.0, -10.0], [0.0, 0.0, 10.0]]], dtype=np.float32)
    for radii in ((1.0, 9.0), (9.0, 1.0)):
        state = VesselState(points, np.array([[0, 1]]), np.asarray(radii), np.array([1.0, 1.0]))
        image = VesselAttenuation(device="cpu").line_integral(
            state, _projection(), width=DETECTOR_PX, height=DETECTOR_PX
        )
        # The central ray runs the segment's whole length, end plane to end plane.
        assert image[0, CENTER_ROW, _column_of(0.0)] == pytest.approx(20.0, abs=0.1)
        assert np.isfinite(image).all()
        assert image.min() >= 0.0


def test_attenuation_varying_along_a_segment_is_integrated_at_the_chord_midpoint() -> None:
    # mu runs 0 -> 2 over x in [-10, 10], so the chord at volume x carries mu = 1 + x / 10.
    image = _chord_image((4.0, 4.0), mus=(0.0, 2.0))
    chord = _chord_through(4.0)
    for volume_x in (-5.0, 0.0, 5.0):
        # The sampled column is half a pixel off, so mu is read where the ray actually is.
        axial_x = (_column_of(volume_x) - CENTER_COLUMN) / MAGNIFICATION
        assert image[0, CENTER_ROW, _column_of(volume_x)] == pytest.approx(chord * (1.0 + axial_x / 10.0), abs=0.05)


def test_environments_do_not_bleed_into_each_other() -> None:
    points = np.repeat(np.array([[[-10.0, 0.0, 0.0], [10.0, 0.0, 0.0]]], dtype=np.float32), 2, axis=0)
    state = VesselState(
        points,
        np.array([[0, 1]]),
        np.array([[4.0, 4.0], [1.0, 1.0]]),
        np.array([[1.0, 1.0], [1.0, 1.0]]),
    )
    image = VesselAttenuation(device="cpu").line_integral(
        state, _projection(), width=DETECTOR_PX, height=DETECTOR_PX
    )
    assert image[0, CENTER_ROW, _column_of(0.0)] == pytest.approx(_chord_through(4.0), abs=0.02)
    assert image[1, CENTER_ROW, _column_of(0.0)] == pytest.approx(_chord_through(1.0), abs=0.02)


# ---------------------------------------------------------------- renderer


def _renderer(patient: PatientVolume, **kwargs) -> SlangFluoroscopyRenderer:
    centre = patient.volume_mm_to_world(patient.center_xyz_mm)
    carm = CArmState(
        source_world_m=np.asarray([centre + [0.0, 0.0, -0.10]]),
        detector_center_world_m=np.asarray([centre + [0.0, 0.0, 0.10]]),
        detector_x_axis_world=np.asarray([[1.0, 0.0, 0.0]]),
        detector_size_m=(0.064, 0.064),
    )
    return SlangFluoroscopyRenderer(
        patient, carm, width=64, height=64, step_mm=1.0, device_type="vulkan",
        catheter_attenuation=False, **kwargs
    )


def _catheter(patient: PatientVolume) -> CatheterState:
    centre = patient.volume_mm_to_world(patient.center_xyz_mm)
    return CatheterState(
        positions_world_m=np.asarray([[centre + [-0.005, 0.0, 0.0], centre + [0.005, 0.0, 0.0]]]),
        valid_nodes=np.asarray([2]),
    )


def test_the_carve_leaves_the_rendered_volume_without_its_lumen(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    renderer = _renderer(patient, vessel_contrast=True)
    assert renderer._carve is not None
    assert renderer._carve.carved_voxels > 0
    assert renderer._carve.contrast_mu_per_mm == pytest.approx(LUMEN_MU - TISSUE_MU, rel=0.15)
    # Coefficients are per node, so on a lumen that really is uniform every node has to land on
    # the same excess. On a real scan they differ, which is what the residual at rest is made of.
    assert renderer._rest_vessel is not None
    np.testing.assert_allclose(
        renderer._rest_vessel.mu_per_mm, LUMEN_MU - TISSUE_MU, rtol=0.15
    )


def test_rendering_without_a_vessel_state_falls_back_to_the_scan_itself(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    frames = _renderer(patient, vessel_contrast=True).render(_catheter(patient))
    for key in ("rgb", "guidance", "dsa", "dsa_guidance"):
        assert frames[key].shape == (1, 64, 64, 3)
    assert not torch_has_nan(frames["attenuation"])


def torch_has_nan(tensor) -> bool:
    return bool(tensor.isnan().any())


def test_moving_the_vessel_shows_up_far_more_in_the_subtracted_channel(tmp_path) -> None:
    """The reason DSA is a subtraction and not a brightness boost.

    The carve leaves a static residual, because a centerline tube cannot describe a real lumen
    exactly. That residual is in the mask and in the live frame identically, so it divides out of
    the subtraction and survives in the plain view -- which is where it would otherwise sit on top
    of the very displacement the deformation is supposed to show.
    """
    patient = _tube_patient(tmp_path)
    renderer = _renderer(patient, vessel_contrast=True, dsa=True)
    rest = renderer._rest_vessel
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


def test_contrast_rendering_needs_a_centerline(tmp_path) -> None:
    patient = _tube_patient(tmp_path, with_centerline=False)
    with pytest.raises(ValueError, match="centerline"):
        _renderer(patient, vessel_contrast=True)


def test_a_vessel_state_for_the_wrong_batch_is_refused(tmp_path) -> None:
    patient = _tube_patient(tmp_path)
    renderer = _renderer(patient, vessel_contrast=True)
    rest = renderer._rest_vessel
    three = VesselState(
        np.repeat(rest.points_mm, 3, axis=0),
        rest.edges,
        np.repeat(rest.radii_mm, 3, axis=0),
        np.repeat(rest.mu_per_mm, 3, axis=0),
    )
    with pytest.raises(ValueError, match="covers 3 environments"):
        renderer.render(_catheter(patient), vessel=three)


def test_vessel_state_rejects_mismatched_shapes() -> None:
    points = np.zeros((1, 4, 3), dtype=np.float32)
    with pytest.raises(ValueError, match="radii_mm"):
        VesselState(points, np.array([[0, 1]]), np.ones(3), np.ones(4))
    with pytest.raises(ValueError, match="edges index outside"):
        VesselState(points, np.array([[0, 9]]), np.ones(4), np.ones(4))

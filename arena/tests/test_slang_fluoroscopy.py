# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import types

import numpy as np
import pytest
import yaml

from i4h_arena.medical.carm import (
    FOLLOW_TIP_ENV_VAR,
    PAN_LOG_ENV_VAR,
    CArmState,
    ReferenceProjectionCArmStateProvider,
    SceneCArmStateProvider,
    follow_tip_enabled,
    pan_log_seconds,
    panned_isocenter_offsets,
)
from i4h_arena.medical.catheter import CatheterState
from i4h_arena.medical.patient_twin import PatientTwin
from i4h_arena.medical.patient_volume import PatientVolume
from i4h_arena.medical.slang_fluoroscopy import SlangFluoroscopyRenderer, solve_projection_geometry


def _patient_volume(
    tmp_path,
    *,
    with_vessel_mask: bool = False,
    coordinate_frame: str = "DICOM_LPS",
    world_from_lps_m: np.ndarray | None = None,
) -> PatientVolume:
    shape = (4, 4, 4)
    spacing_zyx = (3.0, 2.0, 1.0)
    np.save(tmp_path / "mu_volume.npy", np.ones(shape, dtype=np.float32) * 0.01)
    (tmp_path / "metadata.json").write_text(
        json.dumps({"shape_zyx": list(shape), "spacing_zyx_mm": list(spacing_zyx)}),
        encoding="utf-8",
    )
    artifacts = {
        "attenuation_volume": "mu_volume.npy",
        "volume_metadata": "metadata.json",
    }
    if with_vessel_mask:
        vessel_mask = np.zeros(shape, dtype=np.uint8)
        vessel_mask[:, 1:3, 1:3] = 1
        np.save(tmp_path / "vessel_mask.npy", vessel_mask)
        artifacts["vessel_mask"] = "vessel_mask.npy"
    patient_from_lps = np.diag((-1.0, -1.0, 1.0, 1.0)) if coordinate_frame == "NIFTI_RAS" else np.eye(4)
    world_from_lps_m = np.eye(4) if world_from_lps_m is None else world_from_lps_m
    (tmp_path / "patient_twin.yaml").write_text(
        yaml.safe_dump(
            {
                "schema_version": 1,
                "patient_id": "synthetic",
                "coordinate_frame": coordinate_frame,
                "transforms": {
                    "voxel_to_patient_mm": (patient_from_lps @ np.diag((1.0, 2.0, 3.0, 1.0))).tolist(),
                    "world_from_patient_m": (world_from_lps_m @ patient_from_lps).tolist(),
                },
                "artifacts": artifacts,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    return PatientVolume.load(PatientTwin.load(tmp_path / "patient_twin.yaml"))


def _carm(patient: PatientVolume) -> CArmState:
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    return CArmState(
        source_world_m=np.asarray([center + [0.0, 0.0, -0.10]]),
        detector_center_world_m=np.asarray([center + [0.0, 0.0, 0.10]]),
        detector_x_axis_world=np.asarray([[1.0, 0.0, 0.0]]),
        detector_size_m=(0.064, 0.064),
    )


def test_scene_carm_provider_reads_isaac_xyzw_quaternions() -> None:
    class FakeAsset:
        def __init__(self, position, quaternion):
            self.position = np.asarray([position], dtype=np.float32)
            self.quaternion = np.asarray([quaternion], dtype=np.float32)

        def get_world_poses(self):
            return self.position, self.quaternion

    source = FakeAsset((0.0, 0.0, -1.0), (0.0, 0.0, 0.0, 1.0))
    detector = FakeAsset((0.0, 0.0, 1.0), (0.0, 0.0, np.sqrt(0.5), np.sqrt(0.5)))

    state = SceneCArmStateProvider(source, detector, detector_size_m=(0.2, 0.2)).snapshot(1)

    np.testing.assert_allclose(state.detector_x_axis_world, [[0.0, 1.0, 0.0]], atol=1e-6)


@pytest.mark.parametrize("coordinate_frame", ["DICOM_LPS", "NIFTI_RAS"])
def test_reference_projection_provider_aims_the_named_views_at_the_patient(tmp_path, coordinate_frame) -> None:
    """AP must run anterior-posterior, and the obliques must sweep about the body.

    The frame used to be taken from the volume's storage axes. On a real twin
    those map to patient left, posterior and superior, which aimed "AP" along the
    patient's *superior* axis -- a projection down the length of the body -- and
    swept the orbit toward the left instead of around it, leaving a true AP
    unreachable at any angle.

    Asserting the beam in patient terms is what catches that. The test this
    replaced compared the projection against the volume frame and passed
    throughout, because agreeing with the volume frame was the bug.

    Both fixture conventions put patient left, posterior and superior along
    the positive world axes.
    """
    patient = _patient_volume(tmp_path, coordinate_frame=coordinate_frame)
    left = np.array([1.0, 0.0, 0.0])
    posterior = np.array([0.0, 1.0, 0.0])
    superior = np.array([0.0, 0.0, 1.0])
    anterior = -posterior

    class Orbit:
        angle_rad = np.zeros(1, dtype=np.float32)

    orbit = Orbit()
    provider = ReferenceProjectionCArmStateProvider(
        patient,
        orbit,
        detector_size_m=(0.6144, 0.6144),
    )

    for angle_deg in (0.0, 45.0, 90.0, -30.0):
        orbit.angle_rad[:] = np.deg2rad(angle_deg)
        state = provider.snapshot(1)
        beam = state.detector_center_world_m[0] - state.source_world_m[0]
        beam /= np.linalg.norm(beam)
        radians = np.deg2rad(angle_deg)
        expected = np.sin(radians) * left + np.cos(radians) * anterior

        np.testing.assert_allclose(beam, expected, atol=1e-6)
        # The defining property of an LAO/RAO sweep: it turns about the body's
        # long axis, so the beam stays in the patient's transverse plane.
        assert abs(float(np.dot(beam, superior))) < 1e-6

        projection = solve_projection_geometry(patient, state, width=1024, height=1024)
        np.testing.assert_allclose(projection.translation_xyz_mm, [0.0, 0.0, 0.0], atol=1e-7)

    orbit.angle_rad[:] = 0.0
    state = provider.snapshot(1)
    # Vertical axis of the image, derived by the renderer as beam x u. Inferior
    # means the head is drawn at the top rather than upside down.
    vertical = np.cross(anterior, state.detector_x_axis_world[0])
    np.testing.assert_allclose(vertical, -superior, atol=1e-6)


@pytest.mark.parametrize("rotated", [False, True], ids=["identity", "rotated-and-translated"])
def test_lps_and_ras_produce_identical_carm_poses_and_renderer_geometry(tmp_path, rotated) -> None:
    world_from_lps_m = np.eye(4)
    if rotated:
        from scipy.spatial.transform import Rotation

        world_from_lps_m[:3, :3] = Rotation.from_euler("xyz", [23.0, -37.0, 61.0], degrees=True).as_matrix()
        world_from_lps_m[:3, 3] = [0.4, -0.2, 1.3]
    patients = []
    states = []
    angles = np.deg2rad([0.0, 45.0, 90.0, -30.0])
    for frame in ("DICOM_LPS", "NIFTI_RAS"):
        directory = tmp_path / frame
        directory.mkdir()
        patient = _patient_volume(directory, coordinate_frame=frame, world_from_lps_m=world_from_lps_m)
        original_transform = patient.twin.world_from_patient_m.copy()
        provider = ReferenceProjectionCArmStateProvider(
            patient, types.SimpleNamespace(angle_rad=angles), detector_size_m=(0.6144, 0.6144)
        )
        states.append(provider.snapshot(len(angles)))
        patients.append(patient)
        np.testing.assert_array_equal(patient.twin.world_from_patient_m, original_transform)

    np.testing.assert_allclose(patients[0].volume_xyz_mm_to_world_m, patients[1].volume_xyz_mm_to_world_m)
    for name in ("source_world_m", "detector_center_world_m", "detector_x_axis_world"):
        np.testing.assert_allclose(getattr(states[0], name), getattr(states[1], name), atol=1e-12)
    for index in range(len(angles)):
        projections = [
            solve_projection_geometry(patient, state, width=64, height=64, env_index=index)
            for patient, state in zip(patients, states, strict=True)
        ]
        for name in (
            "rotation_zxy_rad",
            "translation_xyz_mm",
            "local_to_volume",
            "isocenter_volume_mm",
            "source_to_detector_mm",
            "source_to_isocenter_mm",
            "pixel_spacing_mm",
        ):
            np.testing.assert_allclose(getattr(projections[0], name), getattr(projections[1], name), atol=1e-9)


# --------------------------------------------------------------------------- #
# Panning the isocenter to keep the tip in frame
# --------------------------------------------------------------------------- #


def test_a_tip_inside_the_keep_zone_leaves_the_frame_alone():
    """A frame that tracked the tip continuously would shake, and would also
    pin the tip to the middle of every image -- discarding tip position within
    the frame as something a policy could read."""
    pan = panned_isocenter_offsets(
        np.array([0.05]), np.zeros(1), half_fov_m=0.15, keep_fraction=0.6, limits_m=(-1.0, 1.0)
    )

    np.testing.assert_allclose(pan, [0.0])


def test_a_tip_past_the_keep_zone_pans_it_back_to_the_boundary():
    """Exactly to the boundary, not to the centre: the smallest move that
    restores the invariant, which is also how a pan reads in a lab."""
    pan = panned_isocenter_offsets(
        np.array([0.12]), np.zeros(1), half_fov_m=0.15, keep_fraction=0.6, limits_m=(-1.0, 1.0)
    )

    # Keep zone is 0.6 * 0.15 = 0.09, so a tip at 0.12 pans by 0.03.
    np.testing.assert_allclose(pan, [0.03])


def test_the_pan_is_hysteretic_so_a_settled_tip_does_not_creep():
    """Once at the boundary the frame must stop, or every step nudges it again."""
    first = panned_isocenter_offsets(
        np.array([0.12]), np.zeros(1), half_fov_m=0.15, keep_fraction=0.6, limits_m=(-1.0, 1.0)
    )
    second = panned_isocenter_offsets(np.array([0.12]), first, half_fov_m=0.15, keep_fraction=0.6, limits_m=(-1.0, 1.0))

    np.testing.assert_allclose(second, first)


def test_the_pan_follows_a_tip_retreating_the_other_way():
    pan = panned_isocenter_offsets(
        np.array([-0.12]), np.zeros(1), half_fov_m=0.15, keep_fraction=0.6, limits_m=(-1.0, 1.0)
    )

    np.testing.assert_allclose(pan, [-0.03])


def test_the_pan_stops_at_the_edge_of_the_scanned_volume():
    """Panning past the anatomy renders a blank detector, which reads as a
    broken sensor rather than as the frame having run out of patient."""
    pan = panned_isocenter_offsets(
        np.array([5.0]), np.zeros(1), half_fov_m=0.15, keep_fraction=0.6, limits_m=(-0.2, 0.2)
    )

    np.testing.assert_allclose(pan, [0.2])


def test_a_tip_that_is_nowhere_yet_does_not_drag_the_frame_away():
    """Before Newton finalizes its model the tip reads as infinity."""
    pan = panned_isocenter_offsets(
        np.array([np.inf]), np.array([0.02]), half_fov_m=0.15, keep_fraction=0.6, limits_m=(-1.0, 1.0)
    )

    np.testing.assert_allclose(pan, [0.02])


def test_each_environment_pans_on_its_own_tip():
    pan = panned_isocenter_offsets(
        np.array([0.0, 0.12, -0.12]),
        np.zeros(3),
        half_fov_m=0.15,
        keep_fraction=0.6,
        limits_m=(-1.0, 1.0),
    )

    np.testing.assert_allclose(pan, [0.0, 0.03, -0.03])


def test_following_the_tip_is_on_but_can_be_pinned_back(monkeypatch) -> None:
    """On by default because a fixed frame loses the tip for 40% of a run, with
    an escape hatch because panning is the only thing that gives the renderer a
    non-zero pose translation."""
    monkeypatch.delenv(FOLLOW_TIP_ENV_VAR, raising=False)
    assert follow_tip_enabled() is True

    for value in ("1", "true", "yes", "on", "ON"):
        monkeypatch.setenv(FOLLOW_TIP_ENV_VAR, value)
        assert follow_tip_enabled() is True

    for value in ("0", "false", "", "off"):
        monkeypatch.setenv(FOLLOW_TIP_ENV_VAR, value)
        assert follow_tip_enabled() is False


def test_the_pan_log_is_off_until_an_interval_is_asked_for():
    assert pan_log_seconds({}) == 0.0


@pytest.mark.parametrize("value, expected", [("1", 1.0), ("0.5", 0.5), (" 5 ", 5.0)])
def test_the_pan_log_takes_an_interval_in_seconds(value, expected):
    """An interval rather than a frame budget.

    A budget of opening frames spends itself while the tip is still near the
    isocenter and the pan is legitimately zero, so it records only the case it
    was not written to explain, and whether the frame later tracked the tip has
    to be recovered by correlating screenshots.
    """
    assert pan_log_seconds({PAN_LOG_ENV_VAR: value}) == pytest.approx(expected)


@pytest.mark.parametrize("value", ["", "0", "-1", "always"])
def test_an_unusable_pan_setting_just_stays_off(value):
    assert pan_log_seconds({PAN_LOG_ENV_VAR: value}) == 0.0


class _Tip:
    """The slice of the catheter entity the provider reads the tip through."""

    def __init__(self, *positions_world_m) -> None:
        # Trailing particle is the tip, matching positions_world_m's ordering.
        self.data = types.SimpleNamespace(positions_world_m=np.asarray(positions_world_m, dtype=np.float64))


def _orbit_at_zero():
    class Orbit:
        angle_rad = np.zeros(1, dtype=np.float32)

    return Orbit()


def test_without_a_tip_source_the_isocenter_stays_on_the_volume_centre(tmp_path) -> None:
    """The projection tests were all written against a fixed isocenter."""
    patient = _patient_volume(tmp_path)
    provider = ReferenceProjectionCArmStateProvider(patient, _orbit_at_zero(), detector_size_m=(0.6144, 0.6144))

    state = provider.snapshot(1)

    midpoint = 0.5 * (state.source_world_m[0] + state.detector_center_world_m[0])
    np.testing.assert_allclose(midpoint, patient.volume_mm_to_world(patient.center_xyz_mm), atol=1e-9)


def test_the_isocenter_follows_the_tip_along_the_head_foot_axis(tmp_path) -> None:
    """The route runs 510 mm head-foot through a 307 mm field, so without this
    roughly 40% of every episode pairs an action with an image that does not
    contain the tip."""
    patient = _patient_volume(tmp_path)
    centre = patient.volume_mm_to_world(patient.center_xyz_mm)
    # The fixture's world_from_patient_m is the identity, so +z is superior.
    far_superior = centre + np.array([0.0, 0.0, 0.5])
    provider = ReferenceProjectionCArmStateProvider(
        patient,
        _orbit_at_zero(),
        detector_size_m=(0.6144, 0.6144),
        tip_source=_Tip([centre, far_superior]),
    )

    state = provider.snapshot(1)

    midpoint = 0.5 * (state.source_world_m[0] + state.detector_center_world_m[0])
    moved = midpoint - centre
    assert abs(float(moved[2])) > 0.0
    # A table pans along the body, not across it.
    np.testing.assert_allclose(moved[:2], [0.0, 0.0], atol=1e-9)


def test_panning_does_not_disturb_the_beam_direction(tmp_path) -> None:
    """Otherwise following the tip would quietly re-aim the named views, which
    is the bug the anatomical basis was introduced to fix."""
    patient = _patient_volume(tmp_path)
    centre = patient.volume_mm_to_world(patient.center_xyz_mm)
    fixed = ReferenceProjectionCArmStateProvider(patient, _orbit_at_zero(), detector_size_m=(0.6144, 0.6144))
    following = ReferenceProjectionCArmStateProvider(
        patient,
        _orbit_at_zero(),
        detector_size_m=(0.6144, 0.6144),
        tip_source=_Tip([centre, centre + np.array([0.0, 0.0, 0.5])]),
    )

    def beam(provider):
        state = provider.snapshot(1)
        axis = state.detector_center_world_m[0] - state.source_world_m[0]
        return axis / np.linalg.norm(axis)

    np.testing.assert_allclose(beam(following), beam(fixed), atol=1e-9)


def test_a_positive_orbit_angle_is_lao_and_a_negative_one_is_rao(tmp_path) -> None:
    """The presets name a side, so the sign has to put the detector on it.

    ``2 LAO-45`` passes a positive angle and ``4 RAO-30`` a negative one, so a
    flipped sign would silently swap every oblique for its mirror image.
    """
    patient = _patient_volume(tmp_path)
    left = np.array([1.0, 0.0, 0.0])

    class Orbit:
        angle_rad = np.zeros(1, dtype=np.float32)

    orbit = Orbit()
    provider = ReferenceProjectionCArmStateProvider(patient, orbit, detector_size_m=(0.6, 0.6))
    isocenter = patient.volume_mm_to_world(patient.center_xyz_mm)

    orbit.angle_rad[:] = np.deg2rad(45.0)
    lao = provider.snapshot(1).detector_center_world_m[0] - isocenter
    orbit.angle_rad[:] = np.deg2rad(-30.0)
    rao = provider.snapshot(1).detector_center_world_m[0] - isocenter

    assert float(np.dot(lao, left)) > 0.0, "a positive angle must carry the detector to the patient's left"
    assert float(np.dot(rao, left)) < 0.0, "a negative angle must carry the detector to the patient's right"


def test_an_unsupported_patient_frame_is_refused(tmp_path) -> None:
    """A different frame convention would silently reorient every view."""
    from i4h_arena.medical.carm import anatomical_projection_basis

    class Twin:
        coordinate_frame = "RAS"
        world_from_patient_m = np.eye(4)

    with pytest.raises(ValueError, match="coordinate_frame"):
        anatomical_projection_basis(Twin())


def test_patient_volume_preserves_physical_coordinates(tmp_path) -> None:
    patient = _patient_volume(tmp_path)

    np.testing.assert_allclose(patient.center_xyz_mm, [2.0, 4.0, 6.0])
    np.testing.assert_allclose(patient.volume_mm_to_world([[1.0, 2.0, 3.0]]), [[0.001, 0.002, 0.003]])
    np.testing.assert_allclose(patient.world_to_volume_mm([[0.001, 0.002, 0.003]]), [[1.0, 2.0, 3.0]])


def test_projection_geometry_matches_scene_source_and_detector(tmp_path) -> None:
    patient = _patient_volume(tmp_path)
    projection = solve_projection_geometry(patient, _carm(patient), width=64, height=64)

    assert np.isclose(projection.source_to_detector_mm, 200.0)
    assert np.isclose(projection.source_to_isocenter_mm, 100.0)
    assert np.isclose(projection.pixel_spacing_mm, 1.0)
    np.testing.assert_allclose(projection.rotation_zxy_rad, [0.0, 0.0, 0.0], atol=1e-7)
    np.testing.assert_allclose(projection.translation_xyz_mm, [0.0, 0.0, 0.0], atol=1e-7)


def test_reference_viewport_detector_geometry_matches_320_at_half_mm(tmp_path) -> None:
    patient = _patient_volume(tmp_path)
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    carm = CArmState(
        source_world_m=np.asarray([center + [0.0, 0.0, -0.510]]),
        detector_center_world_m=np.asarray([center + [0.0, 0.0, 0.510]]),
        detector_x_axis_world=np.asarray([[1.0, 0.0, 0.0]]),
        detector_size_m=(0.160, 0.160),
    )

    projection = solve_projection_geometry(patient, carm, width=320, height=320)

    assert np.isclose(projection.source_to_detector_mm, 1020.0)
    assert np.isclose(projection.source_to_isocenter_mm, 510.0)
    assert np.isclose(projection.pixel_spacing_mm, 0.5)
    np.testing.assert_allclose(projection.translation_xyz_mm, [0.0, 0.0, 0.0], atol=1e-7)


def test_requested_reference_viewport_geometry_matches_1024_at_point_six_mm(tmp_path) -> None:
    patient = _patient_volume(tmp_path)
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    carm = CArmState(
        source_world_m=np.asarray([center + [0.0, 0.0, -0.510]]),
        detector_center_world_m=np.asarray([center + [0.0, 0.0, 0.510]]),
        detector_x_axis_world=np.asarray([[1.0, 0.0, 0.0]]),
        detector_size_m=(0.6144, 0.6144),
    )

    projection = solve_projection_geometry(patient, carm, width=1024, height=1024)

    assert np.isclose(projection.source_to_detector_mm, 1020.0)
    assert np.isclose(projection.source_to_isocenter_mm, 510.0)
    assert np.isclose(projection.pixel_spacing_mm, 0.6)


def test_slang_adapter_builds_dsa_volume_and_cinematic_frame(tmp_path, monkeypatch) -> None:
    from xray_simulator.rendering import diffdrr_slang_renderer

    captured = {"volumes": [], "configs": []}

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            del spacing, origin_xyz_mm
            captured["volumes"].append(np.asarray(volume).copy())
            captured["configs"].append(cfg)

        def render(self, rotation, translation):
            del rotation, translation
            return np.linspace(0.05, 0.8, 64 * 64, dtype=np.float32).reshape(64, 64)

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)
    patient = _patient_volume(tmp_path, with_vessel_mask=True)
    carm = _carm(patient)
    renderer = SlangFluoroscopyRenderer(
        patient,
        carm,
        width=64,
        height=64,
        step_mm=1.0,
        device_type="vulkan",
        dsa=True,
        dsa_boost=6.0,
        visual_style="cinematic",
    )

    output = renderer.render(CatheterState.empty(1), carm)

    assert len(captured["configs"]) == 2
    assert all(config.normalize is False and config.invert is False for config in captured["configs"])
    assert np.isclose(captured["volumes"][1][:, 1:3, 1:3], 0.06).all()
    assert output["rgb"].shape == (1, 64, 64, 3)
    assert output["dsa"].shape == (1, 64, 64, 3)
    assert output["dsa_guidance"].shape == (1, 64, 64, 3)


def test_slang_adapter_renders_fluoro_polarity_on_a_frame_independent_window(tmp_path, monkeypatch) -> None:
    """Dense anatomy reads dark, and a dense object entering the view leaves the rest alone."""
    from xray_simulator.rendering import diffdrr_slang_renderer

    first_frame = np.full((64, 64), 0.6, dtype=np.float32)
    first_frame[:, 8:16] = 0.02
    second_frame = first_frame.copy()
    second_frame[40:48, 40:48] = 0.001
    frames = iter((first_frame, second_frame))

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            del volume, spacing, origin_xyz_mm, cfg

        def render(self, rotation, translation):
            del rotation, translation
            return next(frames)

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    renderer = SlangFluoroscopyRenderer(patient, carm, width=64, height=64, step_mm=1.0, device_type="vulkan")

    first_rgb = renderer.render(CatheterState.empty(1), carm)["rgb"][0]
    second_rgb = renderer.render(CatheterState.empty(1), carm)["rgb"][0]

    assert first_rgb[0, 12, 0] < first_rgb[0, 40, 0]
    assert second_rgb[32, 32, 0] == first_rgb[32, 32, 0]
    assert second_rgb[44, 44, 0] < second_rgb[32, 32, 0]


def test_switching_to_xray_inverts_the_greys_and_keeps_the_calibrated_window(tmp_path, monkeypatch) -> None:
    """The radiograph look reuses the fitted window, so only the polarity changes."""
    from xray_simulator.rendering import diffdrr_slang_renderer

    frame = np.full((64, 64), 0.6, dtype=np.float32)
    frame[:, 8:16] = 0.02

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            del volume, spacing, origin_xyz_mm, cfg

        def render(self, rotation, translation):
            del rotation, translation
            return frame

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    renderer = SlangFluoroscopyRenderer(patient, carm, width=64, height=64, step_mm=1.0, device_type="vulkan")

    fluoro_rgb = renderer.render(CatheterState.empty(1), carm)["rgb"][0]
    calibrated_window = renderer._display.log_window  # noqa: SLF001
    assert renderer.display_polarity == "fluoro"

    assert renderer.set_display_appearance("xray") == "diagnostic"
    xray_rgb = renderer.render(CatheterState.empty(1), carm)["rgb"][0]

    # A named preset carries its own generic window; taking it would have replaced this one.
    assert renderer._display.log_window == calibrated_window  # noqa: SLF001
    # Dense anatomy read dark under fluoro and reads bright as a radiograph.
    assert fluoro_rgb[0, 12, 0] < fluoro_rgb[0, 40, 0]
    assert xray_rgb[0, 12, 0] > xray_rgb[0, 40, 0]

    assert renderer.set_display_appearance("fluoro") == "fluoro"
    assert renderer.display_polarity == "fluoro"


def test_an_unknown_appearance_is_rejected(tmp_path, monkeypatch) -> None:
    from xray_simulator.rendering import diffdrr_slang_renderer

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            del volume, spacing, origin_xyz_mm, cfg

        def render(self, rotation, translation):
            del rotation, translation
            return np.full((64, 64), 0.5, dtype=np.float32)

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    renderer = SlangFluoroscopyRenderer(patient, carm, width=64, height=64, step_mm=1.0, device_type="vulkan")

    with pytest.raises(ValueError):
        renderer.set_display_appearance("ultrasound")


def _window_renderer(tmp_path, monkeypatch) -> SlangFluoroscopyRenderer:
    from xray_simulator.rendering import diffdrr_slang_renderer

    frame = np.full((64, 64), 0.6, dtype=np.float32)
    frame[:, 8:16] = 0.02

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            del volume, spacing, origin_xyz_mm, cfg

        def render(self, rotation, translation):
            del rotation, translation
            return frame

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    return SlangFluoroscopyRenderer(patient, carm, width=64, height=64, step_mm=1.0, device_type="vulkan")


def test_window_width_narrows_around_the_calibrated_centre(tmp_path, monkeypatch) -> None:
    """Halving the width keeps the midpoint and doubles contrast."""
    renderer = _window_renderer(tmp_path, monkeypatch)
    renderer.render(CatheterState.empty(1), _carm(renderer._patient))  # noqa: SLF001
    low, high = renderer.display_window
    centre = 0.5 * (low + high)

    assert renderer.set_display_control("window_width", 0.5) == 0.5
    narrow_low, narrow_high = renderer.display_window

    assert narrow_high - narrow_low == pytest.approx(0.5 * (high - low))
    assert 0.5 * (narrow_low + narrow_high) == pytest.approx(centre)


def test_window_level_shifts_without_changing_width(tmp_path, monkeypatch) -> None:
    renderer = _window_renderer(tmp_path, monkeypatch)
    renderer.render(CatheterState.empty(1), _carm(renderer._patient))  # noqa: SLF001
    low, high = renderer.display_window
    width = high - low

    renderer.set_display_control("window_level", 0.5)
    shifted_low, shifted_high = renderer.display_window

    assert shifted_high - shifted_low == pytest.approx(width)
    assert shifted_low > low


def test_window_stays_a_valid_line_integral_interval(tmp_path, monkeypatch) -> None:
    """A negative shift cannot push the window below zero, which DisplaySettings rejects."""
    renderer = _window_renderer(tmp_path, monkeypatch)
    renderer.render(CatheterState.empty(1), _carm(renderer._patient))  # noqa: SLF001
    _low, high = renderer.display_window

    renderer.set_display_control("window_level", -1.0)
    shifted_low, shifted_high = renderer.display_window

    assert shifted_low >= 0.0
    assert shifted_low < shifted_high
    # The width survives the clamp rather than collapsing against zero.
    assert shifted_high - shifted_low == pytest.approx(high - _low)


def test_display_controls_are_bounded_and_named(tmp_path, monkeypatch) -> None:
    renderer = _window_renderer(tmp_path, monkeypatch)

    assert renderer.set_display_control("window_width", 99.0) == 4.0
    assert renderer.set_display_control("window_level", -99.0) == -1.0
    with pytest.raises(ValueError, match="unknown display control"):
        renderer.set_display_control("window_depth", 1.0)


def test_recalibrating_refits_the_window_and_clears_adjustments(tmp_path, monkeypatch) -> None:
    renderer = _window_renderer(tmp_path, monkeypatch)
    carm = _carm(renderer._patient)  # noqa: SLF001
    renderer.render(CatheterState.empty(1), carm)
    fitted = renderer.display_window
    renderer.set_display_control("window_width", 0.3)
    assert renderer.display_window != fitted

    renderer.recalibrate_display()
    renderer.render(CatheterState.empty(1), carm)

    assert renderer.display_window == fitted


def _shaft_catheter(patient: PatientVolume) -> CatheterState:
    """A 20 mm shaft lying across the beam through the isocenter."""
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    return CatheterState(
        positions_world_m=np.asarray([[center + [-0.01, 0.0, 0.0], center + [0.01, 0.0, 0.0]]]),
        valid_nodes=np.asarray([2]),
    )


def _renderer_with(patient: PatientVolume, carm: CArmState, **kwargs) -> SlangFluoroscopyRenderer:
    return SlangFluoroscopyRenderer(patient, carm, width=64, height=64, step_mm=1.0, device_type="vulkan", **kwargs)


def _shaft_row(attenuation: np.ndarray, column: int = 20) -> int:
    """Row carrying the most attenuation near the middle of the frame, where the shaft projects.

    Local rather than global: the synthetic backgrounds used here are ramps whose bright end would
    otherwise win.
    """
    return int(np.argmax(attenuation[0, 28:36, column, 0])) + 28


def _ramp_renderer_class(monkeypatch):
    """Patch the upstream renderer with a fixed background so the catheter is the only variable."""
    from xray_simulator.rendering import diffdrr_slang_renderer

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            del volume, spacing, origin_xyz_mm, cfg

        def render(self, rotation, translation):
            del rotation, translation
            return np.linspace(0.3, 0.7, 64 * 64, dtype=np.float32).reshape(64, 64)

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)


def test_a_composited_catheter_attenuates_the_beam_instead_of_being_painted_on(tmp_path, monkeypatch) -> None:
    """Compositing puts the instrument in the beam; painting only reached the finished picture."""
    _ramp_renderer_class(monkeypatch)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    catheter = _shaft_catheter(patient)

    composited = _renderer_with(patient, carm).render(catheter, carm)
    annotated = _renderer_with(patient, carm, catheter_attenuation=False).render(catheter, carm)

    shaft_row = _shaft_row(composited["attenuation"])
    assert shaft_row in (31, 32)
    assert composited["attenuation"][0, shaft_row, 20, 0] > composited["attenuation"][0, 35, 20, 0] + 0.1
    assert composited["rgb"][0, shaft_row, 20, 0] < composited["rgb"][0, 35, 20, 0]

    # Painting leaves a flat fill and never reaches the channel a policy reads.
    assert np.count_nonzero(np.all(annotated["rgb"][0] == 12, axis=-1)) > 0
    assert not np.any(np.all(composited["rgb"][0] == 12, axis=-1))
    assert _shaft_row(annotated["attenuation"]) not in (31, 32)


def test_a_composited_catheter_reaches_the_contrast_run_too(tmp_path, monkeypatch) -> None:
    """The instrument sits in front of both volumes, so one solve has to serve both projections."""
    _ramp_renderer_class(monkeypatch)
    patient = _patient_volume(tmp_path, with_vessel_mask=True)
    carm = _carm(patient)
    catheter = _shaft_catheter(patient)

    output = _renderer_with(patient, carm, dsa=True).render(catheter, carm)

    assert output["dsa"][0, 32, 20, 0] < output["dsa"][0, 40, 20, 0]
    assert not np.any(np.all(output["dsa"][0] == 12, axis=-1))


def test_a_radiopaque_tip_reads_darker_than_the_shaft_behind_it(tmp_path, monkeypatch) -> None:
    _ramp_renderer_class(monkeypatch)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    nodes = center + np.stack((np.linspace(-0.01, 0.01, 21), np.zeros(21), np.zeros(21)), axis=-1)
    catheter = CatheterState(positions_world_m=nodes[None, ...], valid_nodes=np.asarray([21]))

    attenuation = _renderer_with(patient, carm).render(catheter, carm)["attenuation"][0, ..., 0]

    tip_column = int(np.argmax(attenuation[32]))
    assert attenuation[32, tip_column] > attenuation[32, 32] + 0.05


def test_slang_adapter_uses_upstream_renderer_and_composites_catheter(tmp_path, monkeypatch) -> None:
    from xray_simulator.rendering import diffdrr_slang_renderer

    calls = {}

    class FakeRenderer:
        def __init__(self, volume, spacing, origin_xyz_mm=(0.0, 0.0, 0.0), cfg=None):
            calls["shape"] = volume.shape
            calls["spacing"] = spacing
            calls["origin"] = origin_xyz_mm
            calls["config"] = cfg

        def render(self, rotation, translation):
            calls.setdefault("poses", []).append((rotation, translation))
            offset = 0.05 * float(np.linalg.norm(rotation))
            return np.linspace(0.1 + offset, 0.8 + offset, 64 * 64, dtype=np.float32).reshape(64, 64)

    monkeypatch.setattr(diffdrr_slang_renderer, "SlangDiffDRRRenderer", FakeRenderer)
    patient = _patient_volume(tmp_path)
    carm = _carm(patient)
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    catheter = CatheterState(
        positions_world_m=np.asarray([[center + [-0.01, 0.0, 0.0], center + [0.01, 0.0, 0.0]]]),
        valid_nodes=np.asarray([2]),
    )

    renderer = SlangFluoroscopyRenderer(
        patient,
        carm,
        width=64,
        height=64,
        step_mm=1.0,
        device_type="vulkan",
    )
    output = renderer.render(catheter, carm)

    angle = 0.35
    rotation_x = np.array([[1.0, 0.0, 0.0], [0.0, np.cos(angle), -np.sin(angle)], [0.0, np.sin(angle), np.cos(angle)]])
    center = patient.volume_mm_to_world(patient.center_xyz_mm)
    orbit_carm = CArmState(
        source_world_m=np.asarray([center + rotation_x @ np.array([0.0, 0.0, -0.10])]),
        detector_center_world_m=np.asarray([center + rotation_x @ np.array([0.0, 0.0, 0.10])]),
        detector_x_axis_world=np.asarray([[1.0, 0.0, 0.0]]),
        detector_size_m=(0.064, 0.064),
    )
    orbit_output = renderer.render(catheter, orbit_carm)

    assert calls["shape"] == (4, 4, 4)
    assert calls["spacing"] == (3.0, 2.0, 1.0)
    assert calls["config"].device_type == "vulkan"
    assert output["rgb"].shape == (1, 64, 64, 3)
    assert output["guidance"].shape == (1, 64, 64, 3)
    assert output["attenuation"].shape == (1, 64, 64, 1)
    assert _shaft_row(output["attenuation"]) in (31, 32)
    assert np.any(output["guidance"][0, ..., 1] > output["guidance"][0, ..., 0])
    assert not np.array_equal(output["guidance"], output["rgb"])
    assert len(calls["poses"]) == 2
    assert not np.allclose(calls["poses"][0][0], calls["poses"][1][0])
    assert not np.array_equal(output["attenuation"], orbit_output["attenuation"])

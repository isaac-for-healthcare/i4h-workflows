# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from i4h_arena.recording.hdf5 import CAMERA_COMPRESSION, EpisodeRecorder, downsample_frame


class _Frame:
    def __init__(self, value: int) -> None:
        self.value = value

    def to_array(self) -> np.ndarray:
        return np.full((32, 48, 3), self.value, dtype=np.uint8)


class _View:
    def __init__(self) -> None:
        self.step = 0

    def joints(self) -> SimpleNamespace:
        return SimpleNamespace(pos=np.full((1, 6), self.step, dtype=np.float32))

    def camera(self, _name: str) -> _Frame:
        return _Frame(self.step)


def _result(*, succeeded: bool = True) -> SimpleNamespace:
    return SimpleNamespace(
        index=0,
        attempt=1,
        succeeded=succeeded,
        status=SimpleNamespace(value="succeeded" if succeeded else "failed"),
    )


def test_streams_camera_frames_and_commits_episode(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="policy", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    for step in range(70):
        view.step = step
        recorder.on_step(np.full((1, 6), step, dtype=np.float32), view)

    recorder._drain_frames()
    with h5py.File(path, "r") as handle:
        assert handle["data/_attempt/obs/room"].shape == (70, 32, 48, 3)
        assert handle["data/_attempt/obs/room"].chunks == (1, 32, 48, 3)
        assert handle["data/_attempt/obs/room"].compression == "gzip"

    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert "_attempt" not in handle["data"]
        assert handle["data/demo_0/actions"].shape == (70, 6)
        assert handle["data/demo_0/obs/joint_pos"].shape == (70, 6)
        assert handle["data/demo_0/obs/room"].shape == (70, 32, 48, 3)
        assert np.all(handle["data/demo_0/obs/room"][-1] == 69)


class _MedicalView(_View):
    """A sensor that also exposes the pre-display signal, as fluoroscopy does."""

    def sensor_signal(self, _name: str, output: str) -> np.ndarray | None:
        if output != "attenuation":
            return None
        return np.full((32, 48, 1), 0.25 * self.step, dtype=np.float32)


def test_records_the_display_independent_signal_beside_the_image(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("fluoroscopy",))
    view = _MedicalView()

    recorder.begin_episode(0, 1)
    for step in range(4):
        view.step = step
        recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        obs = handle["data/demo_0/obs"]
        assert obs["fluoroscopy"].shape == (4, 32, 48, 3)
        signal = obs["fluoroscopy_attenuation"]
        assert signal.shape == (4, 32, 48, 1)
        assert signal.dtype == np.float32
        # Full precision, not quantized through an 8-bit image.
        assert np.allclose(signal[-1], 0.75)


def test_a_view_without_a_signal_records_images_only(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert list(handle["data/demo_0/obs"]) == ["joint_pos", "room"]


def test_discards_temporary_episode(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="policy", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("room",))
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 6), dtype=np.float32), view)
    recorder.end_episode(_result(succeeded=False), keep=False)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert list(handle["data"]) == []


# --------------------------------------------------------------------------- #
# Per-frame diagnostics
#
# Actions and images cannot tell a clean episode from one where the instrument
# misbehaved but still reached the goal, so a scene may offer measurements of
# the simulation itself. They live outside ``obs`` and are optional.
# --------------------------------------------------------------------------- #
class _DiagnosticView(_View):
    """A scene that measures its own physics, as the catheter does."""

    def __init__(self, *, from_step: int = 0) -> None:
        super().__init__()
        self.from_step = from_step

    def diagnostics(self) -> dict[str, object]:
        if self.step < self.from_step:
            return {}
        return {
            "tip_world_m": np.array([0.1, 0.2, 0.01 * self.step], dtype=np.float32),
            "min_bend_radius_mm": 30.0 - self.step,
        }


def test_diagnostics_are_recorded_beside_the_observations(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _DiagnosticView()

    recorder.begin_episode(0, 1)
    for step in range(5):
        view.step = step
        recorder.on_step(np.zeros((1, 4), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        # Outside obs, so everything walking observations is unaffected.
        assert list(demo["obs"]) == ["joint_pos"]
        assert sorted(demo["diagnostics"]) == ["min_bend_radius_mm", "tip_world_m"]
        # A scalar becomes one column per frame and a vector keeps its width.
        assert demo["diagnostics/min_bend_radius_mm"].shape == (5,)
        assert demo["diagnostics/tip_world_m"].shape == (5, 3)
        assert demo["diagnostics/min_bend_radius_mm"][-1] == 26.0
        np.testing.assert_allclose(demo["diagnostics/tip_world_m"][-1], [0.1, 0.2, 0.04], atol=1e-6)


def test_every_diagnostic_stays_aligned_with_the_actions(tmp_path) -> None:
    """A measurement is only useful if a frame of it means the same frame of
    action, so one that starts partway through is padded rather than trimmed."""
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _DiagnosticView(from_step=3)

    recorder.begin_episode(0, 1)
    for step in range(5):
        view.step = step
        recorder.on_step(np.zeros((1, 4), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        demo = handle["data/demo_0"]
        radius = demo["diagnostics/min_bend_radius_mm"][:]
        assert radius.shape == demo["actions"].shape[:1]
        assert np.isnan(radius[:3]).all()
        assert not np.isnan(radius[3:]).any()
        assert np.isnan(demo["diagnostics/tip_world_m"][:3]).all()


def test_a_scene_with_nothing_to_measure_records_no_diagnostics(tmp_path) -> None:
    """The method is optional, so a scene that omits it gets no group at all
    rather than an empty one readers would have to special-case."""
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="policy", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 4), dtype=np.float32), _View())
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert "diagnostics" not in handle["data/demo_0"]


def test_a_discarded_attempt_does_not_leak_diagnostics_into_the_next(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _DiagnosticView()

    recorder.begin_episode(0, 1)
    for step in range(4):
        view.step = step
        recorder.on_step(np.zeros((1, 4), dtype=np.float32), view)
    recorder.end_episode(_result(succeeded=False), keep=False)

    recorder.begin_episode(1, 1)
    view.step = 50
    recorder.on_step(np.zeros((1, 4), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        radius = handle["data/demo_0/diagnostics/min_bend_radius_mm"][:]
        assert radius.shape == (1,)
        assert radius[0] == -20.0


def test_reset_discards_pre_reset_samples(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow)
    view = _View()

    recorder.begin_episode(0, 1)
    recorder.on_step(np.full((1, 6), 3.0, dtype=np.float32), view)
    recorder.restart_episode(node="drive", task_id="teleop/drive")
    recorder.on_step(np.full((1, 6), 7.0, dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        np.testing.assert_array_equal(handle["data/demo_0/actions"][:], np.full((1, 6), 7.0))
        assert handle["data/demo_0/segments"][0]["start"] == 0


# --------------------------------------------------------------------------- #
# Camera storage
#
# Frames dominate a recording: one 1024x1024 fluoroscopy episode ran to 2 GB
# under lzf, which puts a few hundred demonstrations past any disk we have.
# They are also the policy's only view of the scene, so the saving has to be
# lossless.
# --------------------------------------------------------------------------- #
def _xray_like(rows=96, cols=96, seed=0):
    """A smooth greyscale projection written as three equal channels, which is
    what the fluoroscopy sensor produces and what the filters have to handle."""
    rng = np.random.default_rng(seed)
    row, col = np.mgrid[0:rows, 0:cols]
    field = 120.0 + 90.0 * np.sin(row / 11.0) * np.cos(col / 17.0) + rng.normal(0.0, 3.0, (rows, cols))
    grey = np.clip(field, 0, 255).astype(np.uint8)
    return np.repeat(grey[:, :, np.newaxis], 3, axis=2)


class _XrayView(_View):
    def camera(self, _name: str) -> SimpleNamespace:
        frame = _xray_like(seed=self.step)
        return SimpleNamespace(to_array=lambda: frame)


def _record_frames(path, steps=12):
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("fluoroscopy",))
    view = _XrayView()
    recorder.begin_episode(0, 1)
    for step in range(steps):
        view.step = step
        recorder.on_step(np.zeros((1, 4), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()


def test_camera_frames_survive_compression_unchanged(tmp_path) -> None:
    """The policy trains on these, so a filter that altered a pixel would be
    training on something the simulator never rendered."""
    path = tmp_path / "recording.hdf5"
    _record_frames(path, steps=5)

    with h5py.File(path, "r") as handle:
        stored = handle["data/demo_0/obs/fluoroscopy"][:]

    assert stored.dtype == np.uint8
    for step in range(5):
        np.testing.assert_array_equal(stored[step], _xray_like(seed=step))


def test_a_rendered_projection_actually_compresses(tmp_path) -> None:
    """Guards the choice rather than restating it: lzf is what the recorder
    used, and the point of moving is that it left most of the frame on disk."""
    fast, chosen = tmp_path / "lzf.hdf5", tmp_path / "gzip.hdf5"
    frames = np.stack([_xray_like(seed=step) for step in range(12)])

    for target, filters in ((fast, {"compression": "lzf"}), (chosen, CAMERA_COMPRESSION)):
        with h5py.File(target, "w") as handle:
            handle.create_dataset("obs", data=frames, chunks=(1, *frames.shape[1:]), **filters)

    assert chosen.stat().st_size < 0.7 * fast.stat().st_size


def test_the_frames_are_stored_one_timestep_per_chunk(tmp_path) -> None:
    """A reader wanting one step must not pay to decompress its neighbours."""
    path = tmp_path / "recording.hdf5"
    _record_frames(path, steps=4)

    with h5py.File(path, "r") as handle:
        dataset = handle["data/demo_0/obs/fluoroscopy"]
        assert dataset.chunks == (1, *dataset.shape[1:])
        assert dataset.shuffle is True


# --------------------------------------------------------------------------- #
# Recording resolution
#
# The sensor renders 1024 so the operator can see the wire, but the policy
# manifest asks for 256 and the converter resizes before training. The extra
# pixels are 164 GB per two hundred episodes, thrown away downstream.
# --------------------------------------------------------------------------- #
def test_a_rendered_frame_is_reduced_to_the_recorded_edge():
    reduced = downsample_frame(np.zeros((1024, 1024, 3), dtype=np.uint8), edge=256)

    assert reduced.shape == (256, 256, 3)
    assert reduced.dtype == np.uint8


def test_a_thin_bright_wire_survives_the_reduction():
    """The reason for averaging rather than subsampling: the catheter is about
    a pixel wide at 1024, and taking every fourth pixel drops it out of the
    rows it falls between -- losing the one thing the image is recorded for."""
    frame = np.zeros((1024, 1024, 3), dtype=np.uint8)
    frame[501, :, :] = 255  # a row no 4-stride starting at 0 would sample

    reduced = downsample_frame(frame, edge=256)

    assert reduced[125].max() > 0
    assert frame[::4, ::4, :].max() == 0


def test_the_reduction_does_not_darken_the_image():
    """Casting a float mean back to uint8 truncates, losing half a level on
    every frame; over a dataset that is a systematic brightness shift."""
    frame = np.full((8, 8, 3), 101, dtype=np.uint8)

    assert downsample_frame(frame, edge=4).tolist() == np.full((4, 4, 3), 101).tolist()


def test_a_frame_that_does_not_divide_evenly_is_left_alone():
    """An uneven box would weight the edge pixels differently from the rest."""
    frame = np.zeros((300, 300, 3), dtype=np.uint8)

    assert downsample_frame(frame, edge=256).shape == (300, 300, 3)


def test_a_frame_already_at_the_edge_is_left_alone():
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    assert downsample_frame(frame, edge=256).shape == (256, 256, 3)


def test_the_reduction_can_be_switched_off():
    frame = np.zeros((1024, 1024, 3), dtype=np.uint8)

    assert downsample_frame(frame, edge=0).shape == (1024, 1024, 3)


def test_the_raw_sensor_signal_is_reduced_too():
    """It is queued separately from the viewable image but is the same size,
    and it is float, so leaving it full resolution would cost more than the
    image the change was made for."""
    signal = np.zeros((1024, 1024), dtype=np.float32)

    reduced = downsample_frame(signal, edge=256)

    assert reduced.shape == (256, 256)
    assert reduced.dtype == np.float32


def test_the_recorder_stores_frames_at_the_recorded_edge(tmp_path) -> None:
    path = tmp_path / "recording.hdf5"
    workflow = SimpleNamespace(name="example", mode="teleop", scene="example_scene")
    recorder = EpisodeRecorder(path, workflow=workflow, cameras=("fluoroscopy",), camera_edge=48)
    view = _XrayView()  # renders 96x96

    recorder.begin_episode(0, 1)
    recorder.on_step(np.zeros((1, 4), dtype=np.float32), view)
    recorder.end_episode(_result(), keep=True)
    recorder.close()

    with h5py.File(path, "r") as handle:
        assert handle["data/demo_0/obs/fluoroscopy"].shape == (1, 48, 48, 3)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from types import SimpleNamespace

import h5py
import numpy as np

from i4h_arena.recording.hdf5 import EpisodeRecorder


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
        assert handle["data/_attempt/obs/room"].compression == "lzf"

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

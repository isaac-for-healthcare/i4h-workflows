# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import h5py
import numpy as np
import pytest

from i4h_common.episode import (
    Episode,
    EpisodeError,
    Segment,
    action_path,
    camera_keys,
    demo_names,
    diagnostic_keys,
    episodes,
    read_segments,
    write_segments,
)


def _make(path, *, demos=2, frames=5, legacy=False, cameras=("room",), diagnostics=()):
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            obs = demo.create_group("obs")
            actions = np.arange(frames * 6, dtype=np.float32).reshape(frames, 6)
            if legacy:
                obs.create_dataset("actions", data=actions)
            else:
                demo.create_dataset("actions", data=actions)
            obs.create_dataset("joint_pos", data=np.zeros((frames, 6), dtype=np.float32))
            for camera in cameras:
                obs.create_dataset(camera, data=np.zeros((frames, 4, 4, 3), dtype=np.uint8))
            if diagnostics:
                group = demo.create_group("diagnostics")
                for name in diagnostics:
                    group.create_dataset(name, data=np.arange(frames, dtype=np.float32))
            demo.attrs["success"] = index == 0
            demo.attrs["num_samples"] = frames
    return path


def test_demo_names_numeric_order(tmp_path):
    path = _make(tmp_path / "d.hdf5", demos=12)
    with h5py.File(path) as handle:
        # Lexicographic sorting would put demo_10 before demo_2.
        assert demo_names(handle)[:3] == ["demo_0", "demo_1", "demo_2"]
        assert demo_names(handle)[-1] == "demo_11"


def test_action_path_prefers_top_level(tmp_path):
    path = _make(tmp_path / "d.hdf5")
    with h5py.File(path) as handle:
        assert action_path(handle["data/demo_0"]) == "actions"


def test_action_path_falls_back_to_legacy(tmp_path):
    path = _make(tmp_path / "legacy.hdf5", legacy=True)
    with h5py.File(path) as handle:
        assert action_path(handle["data/demo_0"]) == "obs/actions"


def test_missing_actions_raises(tmp_path):
    path = tmp_path / "bad.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("data").create_group("demo_0")
    with h5py.File(path) as handle, pytest.raises(EpisodeError, match="expected 'actions'"):
        action_path(handle["data/demo_0"])


def test_camera_keys_only_uint8_4d(tmp_path):
    path = _make(tmp_path / "d.hdf5", cameras=("room", "wrist"))
    with h5py.File(path) as handle:
        # joint_pos is float 2-D and must not be mistaken for a camera.
        assert camera_keys(handle["data/demo_0"]) == ["room", "wrist"]


def test_episode_accessors(tmp_path):
    path = _make(tmp_path / "d.hdf5", frames=7)
    with h5py.File(path) as handle:
        first, second = episodes(handle)
        assert first.success is True
        assert second.success is False
        assert first.num_samples == 7
        assert first.actions.shape == (7, 6)
        assert first.states.shape == (7, 6)
        assert first.camera("room").shape == (7, 4, 4, 3)


def test_segments_roundtrip(tmp_path):
    path = _make(tmp_path / "d.hdf5", frames=10)
    segments = [
        Segment("locate", "basic/locate", 0, 2),
        Segment("grasp", "basic/grasp", 2, 6),
        Segment("lift", "basic/lift", 6, 10),
    ]
    with h5py.File(path, "a") as handle:
        write_segments(handle["data/demo_0"], segments)
    with h5py.File(path) as handle:
        restored = read_segments(handle["data/demo_0"])
        assert [s.node for s in restored] == ["locate", "grasp", "lift"]
        assert restored[1].length == 4
        assert Episode("demo_0", handle["data/demo_0"]).segment("grasp").start == 2


def test_segments_absent_is_not_an_error(tmp_path):
    path = _make(tmp_path / "d.hdf5")
    with h5py.File(path) as handle:
        assert read_segments(handle["data/demo_0"]) == ()


def test_write_segments_replaces(tmp_path):
    path = _make(tmp_path / "d.hdf5", frames=4)
    with h5py.File(path, "a") as handle:
        demo = handle["data/demo_0"]
        write_segments(demo, [Segment("a", "basic/a", 0, 4)])
        write_segments(demo, [Segment("b", "basic/b", 0, 4)])
        assert [s.node for s in read_segments(demo)] == ["b"]


def test_validate_rejects_out_of_range_segment(tmp_path):
    path = _make(tmp_path / "d.hdf5", frames=4)
    with h5py.File(path, "a") as handle:
        demo = handle["data/demo_0"]
        write_segments(demo, [Segment("a", "basic/a", 0, 99)])
        with pytest.raises(EpisodeError, match="out of"):
            Episode("demo_0", demo).validate()


def test_validate_accepts_well_formed(tmp_path):
    path = _make(tmp_path / "d.hdf5", frames=4)
    with h5py.File(path, "a") as handle:
        demo = handle["data/demo_0"]
        write_segments(demo, [Segment("a", "basic/a", 0, 4)])
        Episode("demo_0", demo).validate()


# --------------------------------------------------------------------------- #
# Diagnostics
#
# Per-frame measurements of the simulation rather than observations. Optional
# and scene-specific, so every reader has to cope with their absence.
# --------------------------------------------------------------------------- #
def test_diagnostics_are_listed_and_readable(tmp_path):
    path = _make(tmp_path / "d.hdf5", frames=5, diagnostics=("arc_excess_mm", "min_bend_radius_mm"))
    with h5py.File(path) as handle:
        demo = handle["data/demo_0"]

        assert diagnostic_keys(demo) == ["arc_excess_mm", "min_bend_radius_mm"]
        np.testing.assert_array_equal(Episode("demo_0", demo).diagnostic("arc_excess_mm"), np.arange(5))


def test_a_recording_without_diagnostics_reads_as_empty(tmp_path):
    """Every recording made before diagnostics existed has to stay readable."""
    path = _make(tmp_path / "d.hdf5")
    with h5py.File(path) as handle:
        demo = handle["data/demo_0"]

        assert diagnostic_keys(demo) == []
        assert Episode("demo_0", demo).diagnostics == []


def test_diagnostics_are_not_mistaken_for_cameras(tmp_path):
    """``camera_keys`` drives dataset conversion, so anything it picks up would
    be turned into video."""
    path = _make(tmp_path / "d.hdf5", cameras=("room",), diagnostics=("tip_target_distance_m",))
    with h5py.File(path) as handle:
        assert camera_keys(handle["data/demo_0"]) == ["room"]


def test_asking_for_a_diagnostic_that_was_not_recorded_says_what_was(tmp_path):
    path = _make(tmp_path / "d.hdf5", diagnostics=("arc_excess_mm",))
    with h5py.File(path) as handle:
        episode = Episode("demo_0", handle["data/demo_0"])

        with pytest.raises(EpisodeError, match="arc_excess_mm"):
            episode.diagnostic("worst_penetration_mm")


def test_validate_rejects_a_diagnostic_that_does_not_span_the_episode(tmp_path):
    """A short column means frame N of the measurement is not frame N of the
    action, which is the only reason to record them together."""
    path = _make(tmp_path / "d.hdf5", frames=5)
    with h5py.File(path, "a") as handle:
        demo = handle["data/demo_0"]
        demo.create_group("diagnostics").create_dataset("arc_excess_mm", data=np.zeros(3, dtype=np.float32))

        with pytest.raises(EpisodeError, match="3 frames, actions have 5"):
            Episode("demo_0", demo).validate()

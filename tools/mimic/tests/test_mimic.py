# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import h5py
import numpy as np
import pytest

from i4h_common.episode import Segment, demo_names, read_segments, write_segments
from i4h_tools.mimic.cli import expand


def _recording(path, *, demos=2, frames=10, segments=True, successes=(True, False)):
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=np.zeros((frames, 6), dtype=np.float32))
            demo.create_group("obs").create_dataset("joint_pos", data=np.zeros((frames, 6), dtype=np.float32))
            demo.attrs["success"] = successes[index % len(successes)]
            demo.attrs["num_samples"] = frames
            if segments:
                write_segments(
                    demo,
                    [
                        Segment("locate", "basic/locate", 0, 2),
                        Segment("grasp", "basic/grasp", 2, 6),
                        Segment("lift", "basic/lift", 6, frames),
                    ],
                )
    return path


def test_expand_writes_requested_count(tmp_path):
    src = _recording(tmp_path / "in.hdf5")
    written = expand(src, tmp_path / "out.hdf5", episodes=5, noise=0.01)
    assert written == 5
    with h5py.File(tmp_path / "out.hdf5") as handle:
        assert len(demo_names(handle)) == 5


def test_expand_applies_noise(tmp_path):
    src = _recording(tmp_path / "in.hdf5")
    expand(src, tmp_path / "out.hdf5", episodes=1, noise=0.5, seed=1)
    with h5py.File(tmp_path / "out.hdf5") as handle:
        actions = handle["data/demo_0/actions"][()]
    assert np.abs(actions).max() > 0.0


def test_noise_is_deterministic_for_a_seed(tmp_path):
    src = _recording(tmp_path / "in.hdf5")
    expand(src, tmp_path / "a.hdf5", episodes=2, noise=0.1, seed=7)
    expand(src, tmp_path / "b.hdf5", episodes=2, noise=0.1, seed=7)
    with h5py.File(tmp_path / "a.hdf5") as a, h5py.File(tmp_path / "b.hdf5") as b:
        assert np.array_equal(a["data/demo_0/actions"][()], b["data/demo_0/actions"][()])


def test_node_scoped_noise_leaves_other_frames_untouched(tmp_path):
    # The capability node tagging exists for: jitter the grasp only.
    src = _recording(tmp_path / "in.hdf5")
    expand(src, tmp_path / "out.hdf5", episodes=1, noise=0.5, seed=3, node="grasp")
    with h5py.File(tmp_path / "out.hdf5") as handle:
        actions = handle["data/demo_0/actions"][()]
    assert np.all(actions[0:2] == 0.0)  # locate untouched
    assert np.any(actions[2:6] != 0.0)  # grasp jittered
    assert np.all(actions[6:] == 0.0)  # lift untouched


def test_node_scope_records_what_it_touched(tmp_path):
    src = _recording(tmp_path / "in.hdf5")
    expand(src, tmp_path / "out.hdf5", episodes=1, noise=0.1, node="grasp")
    with h5py.File(tmp_path / "out.hdf5") as handle:
        assert handle["data/demo_0"].attrs["mimic_frames_jittered"] == 4
        assert handle["data/demo_0"].attrs["mimic_node"] == "grasp"


def test_unknown_node_is_an_actionable_error(tmp_path):
    src = _recording(tmp_path / "in.hdf5")
    with pytest.raises(ValueError, match="no demo carries a segment"):
        expand(src, tmp_path / "out.hdf5", episodes=1, node="grsap")


def test_untagged_recording_rejects_node_scope(tmp_path):
    src = _recording(tmp_path / "in.hdf5", segments=False)
    with pytest.raises(ValueError, match="predate node tagging|no demo carries"):
        expand(src, tmp_path / "out.hdf5", episodes=1, node="grasp")


def test_segments_survive_expansion(tmp_path):
    src = _recording(tmp_path / "in.hdf5")
    expand(src, tmp_path / "out.hdf5", episodes=1, noise=0.01)
    with h5py.File(tmp_path / "out.hdf5") as handle:
        assert [s.node for s in read_segments(handle["data/demo_0"])] == ["locate", "grasp", "lift"]


def test_include_source_prepends_originals(tmp_path):
    src = _recording(tmp_path / "in.hdf5", demos=2)
    written = expand(src, tmp_path / "out.hdf5", episodes=3, include_source=True)
    assert written == 5
    with h5py.File(tmp_path / "out.hdf5") as handle:
        assert "source_demo" not in handle["data/demo_0"].attrs
        assert handle["data/demo_2"].attrs["source_demo"] == "demo_0"


def test_successful_only_filters(tmp_path):
    src = _recording(tmp_path / "in.hdf5", demos=2, successes=(True, False))
    expand(src, tmp_path / "out.hdf5", episodes=4, successful_only=True)
    with h5py.File(tmp_path / "out.hdf5") as handle:
        assert {handle[f"data/demo_{i}"].attrs["source_demo"] for i in range(4)} == {"demo_0"}


def test_no_successful_demos_is_an_error(tmp_path):
    src = _recording(tmp_path / "in.hdf5", demos=1, successes=(False,))
    with pytest.raises(ValueError, match="no successful demos"):
        expand(src, tmp_path / "out.hdf5", episodes=1, successful_only=True)


def test_empty_recording_is_an_error(tmp_path):
    path = tmp_path / "empty.hdf5"
    with h5py.File(path, "w") as handle:
        handle.create_group("data")
    with pytest.raises(ValueError, match="no demo_"):
        expand(path, tmp_path / "out.hdf5", episodes=1)

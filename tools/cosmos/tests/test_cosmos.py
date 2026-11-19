# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import os

import h5py
import imageio.v3 as iio
import numpy as np
import pytest

from i4h_common.episode import demo_names
from i4h_tools.cosmos.cli import export, import_videos

FRAMES = 10


def _recording(path, *, demos=2):
    rng = np.random.default_rng(0)
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for index in range(demos):
            demo = data.create_group(f"demo_{index}")
            demo.create_dataset("actions", data=np.full((FRAMES, 6), index, dtype=np.float32))
            obs = demo.create_group("obs")
            obs.create_dataset("room", data=rng.integers(0, 255, (FRAMES, 32, 32, 3), dtype=np.uint8))
            obs.create_dataset("wrist", data=rng.integers(0, 255, (FRAMES, 32, 32, 3), dtype=np.uint8))
            demo.attrs["success"] = True
            demo.attrs["num_samples"] = FRAMES
    return path


@pytest.fixture
def clips(tmp_path):
    source = _recording(tmp_path / "in.hdf5")
    videodir = tmp_path / "mp4"
    assert export(source, videodir, camera="room") == 2
    return source, videodir


def _shorten(clip):
    iio.imwrite(str(clip), iio.imread(str(clip))[:-1], fps=30, codec="libx264")


def _leftovers(directory):
    return sorted(p.name for p in directory.iterdir() if p.name.endswith(".tmp"))


def test_import_replaces_only_the_camera(clips, tmp_path):
    source, videodir = clips
    target = tmp_path / "aug.hdf5"
    assert import_videos(source, videodir, target, camera="room") == 2
    with h5py.File(source) as src, h5py.File(target) as dst:
        assert demo_names(dst) == ["demo_0", "demo_1"]
        for name in demo_names(dst):
            demo = dst["data"][name]
            origin = src["data"][demo.attrs["cosmos_source"]]
            np.testing.assert_array_equal(demo["actions"][()], origin["actions"][()])
            np.testing.assert_array_equal(demo["obs/wrist"][()], origin["obs/wrist"][()])
            assert demo["obs/room"].shape == origin["obs/room"].shape
    assert _leftovers(tmp_path) == []


def test_import_output_mode_follows_the_umask(clips, tmp_path):
    source, videodir = clips
    target = tmp_path / "aug.hdf5"
    import_videos(source, videodir, target, camera="room")
    umask = os.umask(0)
    os.umask(umask)
    assert target.stat().st_mode & 0o777 == 0o666 & ~umask


def test_import_overwrite_keeps_the_existing_output_mode(clips, tmp_path):
    source, videodir = clips
    target = tmp_path / "aug.hdf5"
    import_videos(source, videodir, target, camera="room")
    target.chmod(0o600)
    import_videos(source, videodir, target, camera="room")
    assert target.stat().st_mode & 0o777 == 0o600


def test_refused_import_keeps_an_existing_output(clips, tmp_path):
    source, videodir = clips
    target = tmp_path / "aug.hdf5"
    import_videos(source, videodir, target, camera="room")
    before = target.read_bytes()
    _shorten(videodir / "demo_1__room.mp4")
    with pytest.raises(ValueError, match="frames does not match"):
        import_videos(source, videodir, target, camera="room")
    assert target.read_bytes() == before
    assert _leftovers(tmp_path) == []


def test_refused_import_creates_no_output(clips, tmp_path):
    source, videodir = clips
    _shorten(videodir / "demo_0__room.mp4")
    target = tmp_path / "out" / "aug.hdf5"
    with pytest.raises(ValueError, match="frames does not match"):
        import_videos(source, videodir, target, camera="room")
    assert not target.exists()
    assert _leftovers(target.parent) == []

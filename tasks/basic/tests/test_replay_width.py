# SPDX-License-Identifier: Apache-2.0
"""Validate recordings against the destination before a command can be emitted."""

from types import SimpleNamespace

import h5py
import numpy as np
import pytest

from i4h_engine.status import Status
from i4h_tasks.basic.replay.replay import Replay


@pytest.mark.parametrize("width", [3, 4])
def test_replay_checks_destination_width_before_first_action(tmp_path, width):
    recording = tmp_path / "demo.hdf5"
    actions = np.arange(6 * width, dtype=np.float32).reshape(6, width)
    with h5py.File(recording, "w") as stream:
        stream.create_dataset("data/demo_0/actions", data=actions)
    commands = []
    ctx = SimpleNamespace(
        num_envs=1, act=SimpleNamespace(dof=3, set_raw_action=lambda a, robot: commands.append(a.copy()))
    )
    replay = Replay(recording)
    if width == 4:
        with pytest.raises(ValueError, match="expected 3, received 4"):
            replay.on_enter(ctx, None)
        assert replay._actions is None
        assert commands == []
    else:
        replay.on_enter(ctx, None)
        np.testing.assert_array_equal(replay._actions, actions)
        assert commands == []
        for index in range(6):
            assert replay.tick(ctx) is (Status.SUCCESS if index == 5 else Status.RUNNING)
        np.testing.assert_array_equal(np.asarray(commands)[:, 0], actions)

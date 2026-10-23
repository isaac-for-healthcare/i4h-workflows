# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import numpy as np
import pytest

from i4h_common.types import CameraFrame, JointState


def test_joint_state_index_of():
    state = JointState(pos=np.zeros((1, 3)), vel=np.zeros((1, 3)), names=("a", "b", "c"))
    assert state.dof == 3
    assert state.index_of("b") == 1
    with pytest.raises(KeyError):
        state.index_of("z")


def test_joint_state_is_always_batched():
    state = JointState(pos=[0.1, 0.2], vel=[0.0, 0.0])
    assert state.pos.shape == (1, 2)
    assert state.num_envs == 1


def test_camera_frame_decodes_rgb8():
    image = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    frame = CameraFrame(name="room", width=2, height=2, data=image.tobytes())
    np.testing.assert_array_equal(frame.to_array(), image)

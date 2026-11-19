# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The wire layout is the contract between the recording pipeline and deployment."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.tactile.sensor import wire  # noqa: E402


def _frame(joints: np.ndarray | None = None) -> bytes:
    deform = (np.arange(wire.DEFORM_BYTES) % 251).astype(np.uint8).reshape(wire.DEFORM_H, wire.DEFORM_W)
    f6 = np.arange(6, dtype=np.float32)
    contact = np.array([1.0, 2.0, 0.5], dtype=np.float32)
    return wire.pack(7, 42, 1.5, deform, f6, contact, joints)


def test_a_frame_round_trips_without_joints() -> None:
    msg = wire.unpack(_frame())
    assert (msg.channel, msg.frame_id, msg.ts) == (7, 42, 1.5)
    assert msg.deform.shape == (240, 240) and msg.deform.dtype == np.uint8
    np.testing.assert_array_equal(msg.f6, np.arange(6, dtype=np.float32))
    np.testing.assert_array_equal(msg.contact_point, [1.0, 2.0, 0.5])
    assert msg.joints is None
    assert msg.hand_finger == ("left", "middle")


def test_the_joint_trailer_is_detected_by_length_alone() -> None:
    joints = np.linspace(-90.0, 90.0, wire.JOINTS_LEN, dtype=np.float32)
    msg = wire.unpack(_frame(joints))
    np.testing.assert_allclose(msg.joints, joints)
    assert len(_frame(joints)) - len(_frame()) == wire.JOINTS_BYTES


def test_a_wrong_sized_joint_trailer_is_rejected() -> None:
    with pytest.raises(ValueError, match="22"):
        _frame(np.zeros(21, dtype=np.float32))


def test_a_truncated_payload_is_rejected() -> None:
    with pytest.raises(ValueError, match="size mismatch"):
        wire.unpack(_frame()[:-1])


def test_channels_are_right_hand_first_and_thumb_last_within_a_hand() -> None:
    assert [wire.channel_to_hand_finger(c)[0] for c in range(wire.NUM_CHANNELS)] == ["right"] * 5 + ["left"] * 5
    assert wire.channel_to_hand_finger(4) == ("right", "thumb")
    assert wire.channel_to_hand_finger(5) == ("left", "pinky")
    with pytest.raises(ValueError, match="out of range"):
        wire.channel_to_hand_finger(wire.NUM_CHANNELS)


def test_the_header_is_28_bytes() -> None:
    assert wire.HEADER_SIZE == 28

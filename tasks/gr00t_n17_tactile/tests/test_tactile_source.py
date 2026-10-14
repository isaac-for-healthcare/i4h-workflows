# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The live source turns sensor channels into frames in the model's finger order."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.tactile import sharpa  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.tactile.sensor import wire  # noqa: E402


class _StubSubscriber:
    """What the ZMQ subscriber exposes, fed by hand instead of by a socket."""

    def __init__(self, host: str, port: int, force_window_len: int = 16) -> None:
        self.window_len = force_window_len
        self.messages: dict[int, wire.TactileMessage] = {}
        self.joints: dict[str, np.ndarray | None] = {"left": None, "right": None}

    def publish(self, channel: int, value: int) -> None:
        deform = np.full((wire.DEFORM_H, wire.DEFORM_W), value, dtype=np.uint8)
        f6 = np.full(6, float(value), dtype=np.float32)
        self.messages[channel] = wire.TactileMessage(channel, value, float(value), deform, f6, np.zeros(0))

    def snapshot(self):
        out = {"left": {}, "right": {}, "joints": dict(self.joints)}
        for msg in self.messages.values():
            hand, finger = msg.hand_finger
            out[hand][finger] = msg
        return out

    def force_window(self):
        out = {"left": {}, "right": {}}
        for channel in range(wire.NUM_CHANNELS):
            hand, finger = wire.channel_to_hand_finger(channel)
            out[hand][finger] = np.full((self.window_len, 6), float(channel), dtype=np.float32)
        return out

    def close(self) -> None:
        pass


def _source(monkeypatch, window: int = 4) -> tuple[sharpa.SharpaLiveSource, _StubSubscriber]:
    monkeypatch.setattr(sharpa, "SharpaTactileSubscriber", _StubSubscriber)
    source = sharpa.SharpaLiveSource("bridge", 7779, force_window=window)
    return source, source._subscriber


def test_no_frame_until_every_fingertip_has_reported(monkeypatch) -> None:
    source, stub = _source(monkeypatch)
    for channel in range(wire.NUM_CHANNELS - 1):
        stub.publish(channel, channel)
    assert source.latest() is None
    stub.publish(wire.NUM_CHANNELS - 1, wire.NUM_CHANNELS - 1)
    assert source.latest() is not None


def test_fingertips_come_out_in_canonical_order_whatever_the_channels_are(monkeypatch) -> None:
    """Channel 4 is the right thumb and channel 5 the left pinky; the model
    wants left thumb first and right pinky last."""
    source, stub = _source(monkeypatch)
    for channel in range(wire.NUM_CHANNELS):
        stub.publish(channel, channel)
    frame = source.latest()
    assert frame.fingertips == FINGERTIPS
    per_finger = {name: int(frame.deform[i, 0, 0]) for i, name in enumerate(FINGERTIPS)}
    assert per_finger["left_thumb"] == 9
    assert per_finger["left_pinky"] == 5
    assert per_finger["right_thumb"] == 4
    assert per_finger["right_pinky"] == 0
    np.testing.assert_array_equal(frame.force[:, 0], [9, 8, 7, 6, 5, 4, 3, 2, 1, 0])
    assert frame.timestamp == 9.0


def test_hand_joints_arrive_in_radians(monkeypatch) -> None:
    source, stub = _source(monkeypatch)
    for channel in range(wire.NUM_CHANNELS):
        stub.publish(channel, 1)
    stub.joints["left"] = np.full(wire.JOINTS_LEN, 90.0, dtype=np.float32)
    frame = source.latest()
    assert set(frame.hand_joints) == {"left"}
    np.testing.assert_allclose(frame.hand_joints["left"], np.pi / 2)


def test_the_force_window_is_stacked_per_fingertip_and_padded_on_the_left(monkeypatch) -> None:
    source, _stub = _source(monkeypatch, window=4)
    window = source.force_window(6)
    assert window.shape == (len(FINGERTIPS), 6, 6)
    # Left thumb is channel 9; its rows all carry the channel number.
    np.testing.assert_array_equal(window[0], np.full((6, 6), 9.0))
    assert source.force_window(2).shape == (len(FINGERTIPS), 2, 6)

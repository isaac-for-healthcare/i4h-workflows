# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the policy needs from a fingertip tactile sensor, and nothing else.

Nothing in this module knows about SharpaWave, ZMQ or GR00T. The SharpaWave
implementations live in ``sharpa.py``. A different fingertip sensor plugs in by
providing the same three things: a *source* that delivers frames, an *encoder*
that turns one frame into a fixed-width feature per fingertip, and the
*kinematics* that place every fingertip in the hand frame so the model can
anchor each tactile token spatially.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Protocol

import numpy as np

#: Canonical fingertip order. Deform maps, force rows, fingertip poses and the
#: model's per-finger positional embedding are all indexed this way, so a
#: source delivers its fingers in this order whatever the sensor's own channel
#: numbering is.
FINGERTIPS: tuple[str, ...] = (
    "left_thumb",
    "left_index",
    "left_middle",
    "left_ring",
    "left_pinky",
    "right_thumb",
    "right_index",
    "right_middle",
    "right_ring",
    "right_pinky",
)

HANDS: tuple[str, ...] = ("left", "right")

#: Pixels of one deformation map.
DEFORM_SHAPE: tuple[int, int] = (240, 240)
#: Force and torque axes per fingertip.
FORCE_AXES = 6
#: Joints per SharpaWave hand, the input of the fingertip kinematics.
HAND_DOF = 22
#: A fingertip pose as the model consumes it: position (3) and the flattened
#: rotation matrix (9) in the hand's base frame.
POSE_DIM = 12


@dataclass(frozen=True)
class TactileFrame:
    """One synchronized reading of every fingertip."""

    #: Sensor timestamp of the newest fingertip in the frame, seconds.
    timestamp: float
    #: ``(N, 240, 240)`` uint8, ``N == len(fingertips)``.
    deform: np.ndarray
    #: ``(N, 6)`` float32 force and torque.
    force: np.ndarray
    #: Hand name to ``(22,)`` joint angles in radians, sampled with the frame, so
    #: the kinematics place the fingertips where they were when the maps were
    #: taken rather than where the robot state says they are now.
    hand_joints: dict[str, np.ndarray] | None = None
    fingertips: tuple[str, ...] = FINGERTIPS


class TactileSource(Protocol):
    """A stream of :class:`TactileFrame`."""

    def latest(self) -> TactileFrame | None:
        """The newest complete frame, or ``None`` before every fingertip has reported."""

    def force_window(self, length: int) -> np.ndarray:
        """``(N, length, 6)`` force history, oldest first, sampled at the training
        rate and left-padded with the oldest row so the shape never depends on
        how long the source has been running."""

    def close(self) -> None: ...


class TactileEncoder(Protocol):
    """Frozen per-fingertip feature extractor."""

    feature_dim: int

    def encode(self, deform: np.ndarray) -> np.ndarray:
        """``(N, 240, 240)`` deform maps to ``(N, feature_dim)`` features."""


class FingertipKinematics(Protocol):
    """Fingertip poses from hand joint angles."""

    def fingertip_poses(self, hand_joints: dict[str, np.ndarray]) -> np.ndarray:
        """Hand name to joint angles (radians) to ``(N, POSE_DIM)`` poses in canonical order."""


#: What the patched ``Gr00tPolicy.get_action`` calls once per denoising step when
#: the server runs with ``tactile_refresh == "per-step"``. It returns the model
#: inputs built from the newest frame, or ``None`` to keep the previous step's.
TactileInputProvider = Callable[[], dict[str, np.ndarray] | None]

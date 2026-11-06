# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire format of the SharpaWave tactile stream.

Vendored from ``teleop/utils/sharpa_tactile_wire.py`` of the public
``binliunls/xr_teleoperate`` fork (pull request #12, "Add tactile capture"),
which is the recording side of the same stream. Keep the two in sync: the
dataset a checkpoint was trained on and the frames it sees at deployment are
decoded by this one layout.

One message is one fingertip's frame. The bridge on the robot publishes ten
messages per sensor tick (five fingers times two hands):

    offset  type       field
    ------  ---------- --------------------------------
    0       uint32     channel        (0-9; 0-4 right hand, 5-9 left hand)
    4       uint32     frame_id       (SDK frame counter)
    8       float64    ts             (SDK hardware timestamp, seconds)
    16      uint32     deform_len     (57600 for a 240x240 uint8 map)
    20      uint32     f6_len         (24 for six float32)
    24      uint32     cp_len         (contact points, multiples of 12)
    28      uint8[]    deform bytes
    +deform_len float32[]  f6 values
    +f6_len     float32[]  contact point triples (x, y, confidence)
    +cp_len     float32[]  OPTIONAL trailer: the owning hand's 22 joint angles in
                       degrees, sampled when the frame was published, so every
                       fingertip ships with the contemporaneous hand pose.
                       Detected by payload length alone; the header is unchanged
                       and frames without it still decode (``joints is None``).
"""

from __future__ import annotations

import struct
from dataclasses import dataclass

import numpy as np

HEADER_FORMAT = "<IIdIII"
HEADER_SIZE = struct.calcsize(HEADER_FORMAT)  # 28

DEFORM_H = 240
DEFORM_W = 240
DEFORM_BYTES = DEFORM_H * DEFORM_W  # 57600
F6_LEN = 6
F6_BYTES = F6_LEN * 4  # 24

JOINTS_LEN = 22
JOINTS_BYTES = JOINTS_LEN * 4  # 88

NUM_CHANNELS = 10

#: Within-hand channel order. The SDK does not document it; this is the order
#: measured on the hardware (a thumb press shows up under the last slot, not the
#: first), and it is what the recording pipeline used, so the dataset and the
#: checkpoint depend on it. ``channel % 5`` indexes this tuple.
FINGER_NAMES = ("pinky", "ring", "middle", "index", "thumb")


def channel_to_hand_finger(channel: int) -> tuple[str, str]:
    """Map an SDK channel (0-9) to ``("left" | "right", finger)``.

    Channels 0-4 belong to the right hand and 5-9 to the left, per the SDK's
    tactile sample; within a hand the order is :data:`FINGER_NAMES`.
    """
    if not 0 <= channel < NUM_CHANNELS:
        raise ValueError(f"channel out of range [0, {NUM_CHANNELS}): {channel}")
    hand = "right" if channel < 5 else "left"
    return hand, FINGER_NAMES[channel % 5]


@dataclass
class TactileMessage:
    channel: int
    frame_id: int
    ts: float
    #: ``(240, 240)`` uint8 when ``deform_len == DEFORM_BYTES``, else one-dimensional.
    deform: np.ndarray
    #: ``(6,)`` float32.
    f6: np.ndarray
    #: ``(3 * N,)`` float32, possibly empty.
    contact_point: np.ndarray
    #: ``(22,)`` float32 degrees, or ``None`` when the publisher sent no trailer.
    joints: np.ndarray | None = None

    @property
    def hand_finger(self) -> tuple[str, str]:
        return channel_to_hand_finger(self.channel)


def pack(
    channel: int,
    frame_id: int,
    ts: float,
    deform: np.ndarray,
    f6: np.ndarray,
    contact_point: np.ndarray,
    joints: np.ndarray | None = None,
) -> bytes:
    """Encode one fingertip frame into a message body.

    ``joints`` (optional) are the owning hand's joint angles in degrees. When
    given they are appended as the fixed-size trailer; omit them to emit the
    deform-only frame.
    """
    deform_bytes = np.ascontiguousarray(deform, dtype=np.uint8).tobytes()
    f6_bytes = np.ascontiguousarray(f6, dtype=np.float32).tobytes()
    cp_bytes = np.ascontiguousarray(contact_point, dtype=np.float32).tobytes()
    header = struct.pack(
        HEADER_FORMAT, int(channel), int(frame_id), float(ts), len(deform_bytes), len(f6_bytes), len(cp_bytes)
    )
    body = header + deform_bytes + f6_bytes + cp_bytes
    if joints is not None:
        joints_arr = np.ascontiguousarray(joints, dtype=np.float32)
        if joints_arr.size != JOINTS_LEN:
            raise ValueError(f"joints must have {JOINTS_LEN} entries, got {joints_arr.size}")
        body += joints_arr.tobytes()
    return body


def unpack(payload: bytes) -> TactileMessage:
    """Decode a message body, with or without the joint trailer."""
    if len(payload) < HEADER_SIZE:
        raise ValueError(f"payload shorter than header: {len(payload)} < {HEADER_SIZE}")
    channel, frame_id, ts, deform_len, f6_len, cp_len = struct.unpack_from(HEADER_FORMAT, payload, 0)
    core = HEADER_SIZE + deform_len + f6_len + cp_len
    if len(payload) == core:
        has_joints = False
    elif len(payload) == core + JOINTS_BYTES:
        has_joints = True
    else:
        raise ValueError(
            f"payload size mismatch: got {len(payload)}, expected {core} (no joints) or {core + JOINTS_BYTES}"
        )
    offset = HEADER_SIZE
    deform = np.frombuffer(payload, dtype=np.uint8, count=deform_len, offset=offset)
    offset += deform_len
    f6 = np.frombuffer(payload, dtype=np.float32, count=f6_len // 4, offset=offset)
    offset += f6_len
    contact_point = np.frombuffer(payload, dtype=np.float32, count=cp_len // 4, offset=offset)
    offset += cp_len
    joints = np.frombuffer(payload, dtype=np.float32, count=JOINTS_LEN, offset=offset) if has_joints else None
    if deform_len == DEFORM_BYTES:
        deform = deform.reshape(DEFORM_H, DEFORM_W)
    return TactileMessage(channel, frame_id, ts, deform, f6, contact_point, joints)

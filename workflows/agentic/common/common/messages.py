# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import itertools
import time as _time
from dataclasses import asdict, dataclass, field
from typing import TypeVar

import msgpack

_request_counter = itertools.count(1)


@dataclass(slots=True)
class CameraStream:
    ts: int = 0
    frame_num: int = 0
    width: int = 0
    height: int = 0
    focal_len: float = 0.0
    data: bytes = b""

    def __post_init__(self) -> None:
        if self.ts == 0:
            self.ts = _time.time_ns()
        self.data = bytes(self.data)


@dataclass(slots=True)
class RobotState:
    ts: int = 0
    run_id: str = ""
    episode_index: int = 0
    attempt_index: int = 0
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    is_running: bool = True

    def __post_init__(self) -> None:
        if self.ts == 0:
            self.ts = _time.time_ns()
        self.run_id = str(self.run_id)
        self.episode_index = int(self.episode_index)
        self.attempt_index = int(self.attempt_index)
        self.joint_positions = list(self.joint_positions)
        self.joint_velocities = list(self.joint_velocities)


@dataclass(slots=True)
class RobotCommand:
    ts: int = 0
    request_id: int = 0
    run_id: str = ""
    episode_index: int = 0
    attempt_index: int = 0
    horizon: int = 1
    dt: float = 0.0
    inference_ts: int = 0
    joint_positions: list[float] = field(default_factory=list)

    def __post_init__(self) -> None:
        if self.request_id == 0:
            self.request_id = next(_request_counter)
        if self.ts == 0:
            self.ts = _time.time_ns()
        self.run_id = str(self.run_id)
        self.episode_index = int(self.episode_index)
        self.attempt_index = int(self.attempt_index)
        self.joint_positions = list(self.joint_positions)


Message = CameraStream | RobotState | RobotCommand
MessageT = TypeVar("MessageT", CameraStream, RobotState, RobotCommand)


def encode_message(message: Message) -> bytes:
    return msgpack.packb(asdict(message), use_bin_type=True)


def _decode_dataclass(payload: bytes, cls: type[MessageT]) -> MessageT:
    data = msgpack.unpackb(payload, raw=False)
    if not isinstance(data, dict):
        raise ValueError(f"expected msgpack map for {cls.__name__}")
    return cls(**data)


def decode_camera(payload: bytes) -> CameraStream:
    return _decode_dataclass(payload, CameraStream)


def decode_state(payload: bytes) -> RobotState:
    return _decode_dataclass(payload, RobotState)


def decode_command(payload: bytes) -> RobotCommand:
    return _decode_dataclass(payload, RobotCommand)

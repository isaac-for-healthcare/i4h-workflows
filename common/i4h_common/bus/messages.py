# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Wire messages, msgpack-encoded.

Only values that cross a process boundary appear here. Workflow edges hand objects
over directly in-process and are never serialized.

Every message carries ``ts`` (ns) and, where a request/response pairing matters,
``seq``. Decoders tolerate unknown keys so a newer backend can add fields
without breaking an older arena.
"""

from __future__ import annotations

import itertools
import time
from dataclasses import asdict, dataclass, field, fields
from typing import Any, TypeVar

import msgpack

_seq = itertools.count(1)


def _now() -> int:
    return time.time_ns()


@dataclass(slots=True)
class Envelope:
    """Fields shared by every message."""

    ts: int = 0
    seq: int = 0

    def __post_init__(self) -> None:
        if self.ts == 0:
            self.ts = _now()
        if self.seq == 0:
            self.seq = next(_seq)


@dataclass(slots=True)
class CameraStream(Envelope):
    name: str = ""
    width: int = 0
    height: int = 0
    encoding: str = "rgb8"
    focal_len: float = 0.0
    frame_num: int = 0
    data: bytes = b""


@dataclass(slots=True)
class RobotState(Envelope):
    run_id: str = ""
    episode_index: int = 0
    node: str = ""
    joint_positions: list[float] = field(default_factory=list)
    joint_velocities: list[float] = field(default_factory=list)
    tcp_pos: list[float] = field(default_factory=list)
    tcp_quat: list[float] = field(default_factory=list)
    is_running: bool = True


@dataclass(slots=True)
class RobotCommand(Envelope):
    run_id: str = ""
    horizon: int = 1
    dt: float = 0.0
    joint_positions: list[float] = field(default_factory=list)


@dataclass(slots=True)
class TaskSpecMsg:
    """Sent once when a remote task node is entered. Starts an inference session."""

    ts: int = 0
    seq: int = 0
    task_uid: str = ""
    task_id: str = ""
    run_id: str = ""
    episode_index: int = 0
    prompt: str = ""
    checkpoint: str = ""
    params: dict[str, Any] = field(default_factory=dict)
    observation: dict[str, Any] = field(default_factory=dict)
    model: dict[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.ts == 0:
            self.ts = _now()
        if self.seq == 0:
            self.seq = next(_seq)


@dataclass(slots=True)
class ObsFrame(Envelope):
    """One observation, shaped by the task manifest's ``observation`` block."""

    task_uid: str = ""
    step: int = 0
    state: list[float] = field(default_factory=list)
    state_names: list[str] = field(default_factory=list)
    images: dict[str, bytes] = field(default_factory=dict)
    image_shapes: dict[str, list[int]] = field(default_factory=dict)


@dataclass(slots=True)
class ActionChunk(Envelope):
    """A horizon of actions produced for one ``ObsFrame``."""

    task_uid: str = ""
    for_step: int = 0
    horizon: int = 1
    dt: float = 0.0
    #: Flattened row-major ``(horizon, dof)``.
    actions: list[float] = field(default_factory=list)
    dof: int = 0

    def reshape(self) -> list[list[float]]:
        if self.dof <= 0:
            raise ValueError("action chunk has dof=0")
        return [self.actions[i * self.dof : (i + 1) * self.dof] for i in range(self.horizon)]


@dataclass(slots=True)
class TaskStatusMsg(Envelope):
    task_uid: str = ""
    #: ``running`` | ``success`` | ``failure`` | ``ready`` | ``error``
    status: str = "running"
    detail: str = ""

    # -- action contract, sent with status="ready" ------------------------
    # The backend owns the model, so it is authoritative about what its action
    # vectors mean. A manifest records what a human wrote and can go stale the
    # moment someone passes --checkpoint; this cannot. The arena side validates
    # the manifest's declaration against this and refuses to run on a mismatch.
    #: ``joint_position`` | ``joint_velocity`` | ``ee_pose``
    action_space: str = ""
    #: ``joints`` | ``pos_quat`` | ``pos_euler`` | ``pos_axis_angle``
    action_layout: str = ""
    #: Width of one action row.
    action_dof: int = 0
    #: Which robots a row addresses, in order. Empty means the scene's default.
    action_robots: list[str] = field(default_factory=list)
    #: ``last`` (final column is the jaw) | ``none``
    action_gripper: str = "none"


@dataclass(slots=True)
class WorkflowEventMsg(Envelope):
    """Workflow telemetry: what the engine just did."""

    run_id: str = ""
    workflow: str = ""
    episode_index: int = 0
    step: int = 0
    #: ``workflow_started`` | ``node_entered`` | ``node_succeeded`` | ``node_failed``
    #: | ``node_aborted`` | ``workflow_finished``
    event: str = ""
    node: str = ""
    task_id: str = ""
    outputs: dict[str, Any] = field(default_factory=dict)
    detail: str = ""


@dataclass(slots=True)
class WorkflowControl(Envelope):
    """External steering of a running workflow."""

    #: ``abort`` | ``pause`` | ``resume`` | ``skip`` | ``set_input``
    command: str = ""
    node: str = ""
    port: str = ""
    value: Any = None


_BY_NAME: dict[str, type] = {
    cls.__name__: cls
    for cls in (
        CameraStream,
        RobotState,
        RobotCommand,
        TaskSpecMsg,
        ObsFrame,
        ActionChunk,
        TaskStatusMsg,
        WorkflowEventMsg,
        WorkflowControl,
    )
}

MessageT = TypeVar("MessageT")


def encode(message: Any) -> bytes:
    """msgpack-encode a message dataclass, tagged with its type name."""
    payload = asdict(message)
    payload["__type__"] = type(message).__name__
    return msgpack.packb(payload, use_bin_type=True)


def decode(payload: bytes, expect: type[MessageT] | None = None) -> MessageT:
    """Decode a message, optionally asserting its type.

    Unknown keys are dropped rather than raising, so a newer producer can add
    fields without breaking an older consumer.
    """
    raw = msgpack.unpackb(bytes(payload), raw=False)
    if not isinstance(raw, dict):
        raise ValueError("expected a msgpack map")
    type_name = raw.pop("__type__", None)
    cls = expect or (_BY_NAME.get(type_name) if type_name else None)
    if cls is None:
        raise ValueError(f"cannot resolve message type from {type_name!r}")
    if expect is not None and type_name is not None and type_name != expect.__name__:
        raise ValueError(f"expected {expect.__name__} on this key, got {type_name}")
    known = {f.name for f in fields(cls)}
    return cls(**{k: v for k, v in raw.items() if k in known})

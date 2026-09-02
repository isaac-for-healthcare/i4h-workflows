# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Plain value types shared across every workflow project.

Everything here is batched over environments: the leading axis of any array is
``num_envs``. A single-env value is shape ``(1, ...)``, never ``(...)``. Keeping
that invariant everywhere removes a whole class of broadcasting bugs at the
task/adapter boundary.

These are numpy, not torch, deliberately: it keeps ``tasks/*`` installable and
testable without a 2 GB torch wheel. The Isaac adapter converts at the boundary.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np


def satisfied(value: Any, *, across: str = "any") -> bool:
    """Collapse a per-env predicate result to a single bool.

    Predicates return a mask over environments, and callers disagree about what
    that should mean: a workflow's ``until=`` fires as soon as *any* env reaches the
    goal, while :class:`~i4h_tasks.basic.control.wait_until.WaitUntil` waits for
    *all* of them. The quantifier is explicit so every call site states which
    meaning it wants.
    """
    if isinstance(value, bool):
        return value
    array = np.asarray(value)
    if not array.size:
        return False
    return bool(array.all()) if across == "all" else bool(array.any())


def as_batch(value: Any, num_envs: int, width: int) -> np.ndarray:
    """Coerce ``value`` to a ``(num_envs, width)`` float32 array.

    Accepts a scalar, a ``(width,)`` vector (broadcast across envs), or an
    already-batched array.
    """
    array = np.asarray(value, dtype=np.float32)
    if array.ndim == 0:
        return np.full((num_envs, width), float(array), dtype=np.float32)
    if array.ndim == 1:
        if array.shape[0] != width:
            raise ValueError(f"expected width {width}, got {array.shape[0]}")
        return np.broadcast_to(array, (num_envs, width)).astype(np.float32, copy=True)
    if array.shape != (num_envs, width):
        raise ValueError(f"expected shape {(num_envs, width)}, got {array.shape}")
    return array.astype(np.float32, copy=False)


@dataclass(frozen=True, slots=True)
class Pose:
    """Batched rigid-body pose: position ``(N, 3)`` + quaternion ``(N, 4)`` in ``wxyz``.

    The common task/policy boundary uses ``wxyz``. Simulator-facing tensors use
    ``xyzw``; the arena adapter converts explicitly.
    """

    pos: np.ndarray
    quat: np.ndarray

    def __post_init__(self) -> None:
        pos = np.atleast_2d(np.asarray(self.pos, dtype=np.float32))
        quat = np.atleast_2d(np.asarray(self.quat, dtype=np.float32))
        if pos.shape[-1] != 3:
            raise ValueError(f"pose position must be (N, 3), got {pos.shape}")
        if quat.shape[-1] != 4:
            raise ValueError(f"pose quaternion must be (N, 4) wxyz, got {quat.shape}")
        if pos.shape[0] != quat.shape[0]:
            raise ValueError(f"pose batch mismatch: pos {pos.shape[0]} vs quat {quat.shape[0]}")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "quat", quat)

    @property
    def num_envs(self) -> int:
        return int(self.pos.shape[0])

    @classmethod
    def identity(cls, num_envs: int = 1) -> Pose:
        return cls(
            pos=np.zeros((num_envs, 3), dtype=np.float32),
            quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_envs, 1)),
        )

    @classmethod
    def from_xyz(cls, x: float, y: float, z: float, num_envs: int = 1) -> Pose:
        return cls(
            pos=np.tile(np.array([x, y, z], dtype=np.float32), (num_envs, 1)),
            quat=np.tile(np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float32), (num_envs, 1)),
        )

    def translated(self, offset: Any) -> Pose:
        """Return this pose shifted by ``offset`` in the world frame."""
        delta = as_batch(offset, self.num_envs, 3)
        return Pose(pos=self.pos + delta, quat=self.quat)

    def distance_to(self, other: Pose) -> np.ndarray:
        """Per-env Euclidean distance between positions, shape ``(N,)``."""
        return np.linalg.norm(self.pos - other.pos, axis=-1)


@dataclass(frozen=True, slots=True)
class JointState:
    """Batched joint positions/velocities, shape ``(N, dof)``, plus joint names."""

    pos: np.ndarray
    vel: np.ndarray
    names: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        pos = np.atleast_2d(np.asarray(self.pos, dtype=np.float32))
        vel = np.atleast_2d(np.asarray(self.vel, dtype=np.float32))
        if pos.shape != vel.shape:
            raise ValueError(f"joint pos/vel shape mismatch: {pos.shape} vs {vel.shape}")
        object.__setattr__(self, "pos", pos)
        object.__setattr__(self, "vel", vel)
        object.__setattr__(self, "names", tuple(self.names))

    @property
    def num_envs(self) -> int:
        return int(self.pos.shape[0])

    @property
    def dof(self) -> int:
        return int(self.pos.shape[1])

    def index_of(self, name: str) -> int:
        try:
            return self.names.index(name)
        except ValueError as exc:
            raise KeyError(f"unknown joint {name!r}; have {list(self.names)}") from exc


@dataclass(frozen=True, slots=True)
class ObjectState:
    """A scene object's pose plus linear and angular velocity."""

    name: str
    pose: Pose
    lin_vel: np.ndarray
    ang_vel: np.ndarray | None = None

    def __post_init__(self) -> None:
        lin_vel = np.atleast_2d(np.asarray(self.lin_vel, dtype=np.float32))
        ang_vel = (
            np.zeros_like(lin_vel)
            if self.ang_vel is None
            else np.atleast_2d(np.asarray(self.ang_vel, dtype=np.float32))
        )
        if lin_vel.shape != (self.pose.num_envs, 3):
            raise ValueError(f"linear velocity must be {(self.pose.num_envs, 3)}, got {lin_vel.shape}")
        if ang_vel.shape != lin_vel.shape:
            raise ValueError(f"angular velocity shape mismatch: {ang_vel.shape} vs {lin_vel.shape}")
        object.__setattr__(self, "lin_vel", lin_vel)
        object.__setattr__(self, "ang_vel", ang_vel)

    @property
    def is_settled(self) -> np.ndarray:
        """Per-env mask: linear speed below 1 cm/s."""
        return np.linalg.norm(self.lin_vel, axis=-1) < 0.01


@dataclass(slots=True)
class CameraFrame:
    """One rendered camera image.

    ``data`` is raw bytes (JPEG or raw RGB per ``encoding``) so the same type
    serves both the in-process path and the zenoh wire without a conversion.
    """

    name: str
    width: int
    height: int
    data: bytes = b""
    encoding: str = "rgb8"
    focal_len: float = 0.0
    frame_num: int = 0
    ts: int = 0
    meta: dict[str, Any] = field(default_factory=dict)

    def to_array(self) -> np.ndarray:
        if self.encoding != "rgb8":
            raise ValueError(f"cannot decode {self.encoding!r} without an image codec")
        return np.frombuffer(self.data, dtype=np.uint8).reshape(self.height, self.width, 3)


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product of batched ``wxyz`` quaternions."""
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    ).astype(np.float32)


def quat_rotate(quat: np.ndarray, vec: np.ndarray) -> np.ndarray:
    """Rotate batched vectors by batched ``wxyz`` quaternions."""
    w = quat[..., 0:1]
    xyz = quat[..., 1:4]
    t = 2.0 * np.cross(xyz, vec)
    return (vec + w * t + np.cross(xyz, t)).astype(np.float32)

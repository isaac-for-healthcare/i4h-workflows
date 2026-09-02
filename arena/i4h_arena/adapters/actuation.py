# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
""":class:`i4h_common.world.Actuation` → an IsaacLab action tensor.

Tasks write here; the runner reads :meth:`ArenaActuation.tensor` once per tick
and hands it to ``env.step``. Nobody else composes actions, which is what makes
"who is driving this joint?" a question with exactly one answer.

For multi-robot scenes (``dual_psm_reach``) each robot owns a slice of the
action vector, so two parallel branches can drive two arms in the same tick —
the engine's conflict detector keys on the robot name, so those writes do not
collide.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import torch

from i4h_common.types import Pose, as_batch, quat_mul
from i4h_common.world import UnsupportedActuation

logger = logging.getLogger("i4h_arena.actuation")


@dataclass(frozen=True, slots=True)
class RobotSlice:
    """Which columns of the action vector belong to one robot."""

    name: str
    start: int
    stop: int
    #: Column of the jaw within this slice, or ``None`` for a gripper-less arm.
    gripper_index: int | None = None
    #: Number of measured joints at the start of this slice.  Whole-body
    #: controllers may append non-joint commands to their action vector.
    joint_width: int | None = None

    @property
    def width(self) -> int:
        return self.stop - self.start


class ArenaActuation:
    """Composes one action tensor per tick from whatever the active nodes wrote."""

    def __init__(
        self,
        *,
        num_envs: int,
        action_dim: int,
        action_space: str = "joint_position",
        device: str = "cpu",
        slices: tuple[RobotSlice, ...] = (),
        initial: np.ndarray | None = None,
        view: object | None = None,
        relative_ee: bool = False,
    ) -> None:
        self._num_envs = num_envs
        self._action_dim = action_dim
        self._action_space = action_space
        self._device = device
        self._slices = slices or (RobotSlice("robot", 0, action_dim, gripper_index=action_dim - 1),)
        self._by_name = {s.name: s for s in self._slices}
        self._view = view
        self._relative_ee = relative_ee
        self._logged_ee_targets = 0
        start = (
            np.zeros((num_envs, action_dim), dtype=np.float32) if initial is None else np.asarray(initial, np.float32)
        )
        self._buffer = start.copy()
        #: Last fully-composed command, so ``hold`` has something to repeat.
        self._previous = self._buffer.copy()

    # -- Actuation -------------------------------------------------------
    @property
    def dof(self) -> int:
        return self._action_dim

    @property
    def action_space(self) -> str:
        return self._action_space

    def _slice(self, robot: str) -> RobotSlice:
        try:
            return self._by_name[robot]
        except KeyError as exc:
            raise KeyError(f"unknown robot {robot!r}; scene declares {list(self._by_name)}") from exc

    def set_joint_targets(self, values: np.ndarray, robot: str = "robot") -> None:
        if self._action_space not in ("joint_position", "joint_velocity"):
            raise UnsupportedActuation(f"scene action_space is {self._action_space!r}; use set_ee_target instead")
        target = self._slice(robot)
        values_array = np.asarray(values)
        input_width = 1 if values_array.ndim == 0 else int(values_array.shape[-1])
        allowed_widths = {target.width}
        if target.joint_width is not None:
            allowed_widths.add(target.joint_width)
        if input_width not in allowed_widths:
            expected = " or ".join(str(width) for width in sorted(allowed_widths))
            raise ValueError(f"expected width {expected}, got {input_width}")
        self._buffer[:, target.start : target.start + input_width] = as_batch(values, self._num_envs, input_width)

    def set_gripper(self, width: np.ndarray | float, robot: str = "robot") -> None:
        target = self._slice(robot)
        if target.gripper_index is None:
            raise UnsupportedActuation(f"robot {robot!r} has no gripper")
        column = target.start + target.gripper_index
        self._buffer[:, column] = np.asarray(as_batch(width, self._num_envs, 1)).reshape(self._num_envs)

    def set_ee_target(self, pose: Pose, robot: str = "robot") -> None:
        if self._action_space != "ee_pose":
            raise UnsupportedActuation(
                f"scene action_space is {self._action_space!r}, so this embodiment has no Cartesian "
                f"target; workflow-lint should have rejected this via requires/provides"
            )
        target = self._slice(robot)
        if self._relative_ee:
            if self._view is None:
                raise RuntimeError("relative Cartesian actuation requires a scene view")
            current = self._view.tcp(robot)
            inverse = current.quat.copy()
            inverse[:, 1:] *= -1.0
            rotation_delta = quat_mul(pose.quat, inverse)
            # IsaacLab's relative pose command is xyz + rotation vector.
            vector = rotation_delta[:, 1:4]
            vector_norm = np.linalg.norm(vector, axis=-1, keepdims=True)
            angle = 2.0 * np.arctan2(vector_norm, np.clip(rotation_delta[:, 0:1], -1.0, 1.0))
            axis_angle = np.divide(
                vector * angle,
                vector_norm,
                out=np.zeros_like(vector),
                where=vector_norm > 1e-8,
            )
            flat = np.concatenate([pose.pos - current.pos, axis_angle], axis=-1).astype(np.float32)
        else:
            # The simulator action boundary uses ``xyzw``; the common layer
            # uses ``wxyz``.
            quat_xyzw = pose.quat[:, [1, 2, 3, 0]]
            flat = np.concatenate([pose.pos, quat_xyzw], axis=-1).astype(np.float32)
        if self._logged_ee_targets < 2:
            logger.debug(
                "Cartesian action %s robot=%s value=%s",
                self._logged_ee_targets + 1,
                robot,
                flat.round(5).tolist(),
            )
            self._logged_ee_targets += 1
        width = min(target.width, flat.shape[-1])
        self._buffer[:, target.start : target.start + width] = flat[:, :width]

    def set_ee_delta(self, values: np.ndarray, robot: str = "robot") -> None:
        if self._action_space != "ee_pose" or not self._relative_ee:
            raise UnsupportedActuation("relative Cartesian commands require a relative ee_pose scene")
        target = self._slice(robot)
        if target.width != 6:
            raise ValueError(f"relative Cartesian action requires width 6, got {target.width}")
        self._buffer[:, target.start : target.stop] = as_batch(values, self._num_envs, 6)

    def hold(self, robot: str = "robot") -> None:
        # Repeating a joint-position target holds position, but repeating a
        # relative Cartesian delta keeps moving the tool.  Zero delta asks the
        # relative IK controller to retain the measured pose.
        if self._relative_ee:
            target = self._slice(robot)
            self._buffer[:, target.start : target.stop] = 0.0
            return
        if robot == "robot" and robot not in self._by_name and len(self._by_name) > 1:
            self._buffer[:, :] = self._previous
            return
        target = self._slice(robot)
        self._buffer[:, target.start : target.stop] = self._previous[:, target.start : target.stop]

    def set_raw_action(self, values: np.ndarray, robot: str = "robot") -> None:
        """Copy a controller-native action into its robot slice."""
        target = self._slice(robot)
        self._buffer[:, target.start : target.stop] = as_batch(values, self._num_envs, target.width)

    # -- runner-facing ---------------------------------------------------
    def tensor(self) -> torch.Tensor:
        """The composed action for this tick."""
        self._previous = self._buffer.copy()
        return torch.from_numpy(self._buffer).to(self._device)

    def numpy(self) -> np.ndarray:
        """The composed action as numpy — what the recorder stores."""
        return self._buffer.copy()

    def seed(self, values: np.ndarray) -> None:
        """Set the buffer directly, e.g. to a home pose before the first tick."""
        # ``values`` may be the SceneView's cached home-joint array.  Keep the
        # command buffer independent: tasks read that home reference again when
        # resolving relative targets, while every actuator write mutates this
        # buffer in place.
        self._buffer = np.array(
            as_batch(values, self._num_envs, self._action_dim),
            dtype=np.float32,
            copy=True,
        )
        self._previous = self._buffer.copy()

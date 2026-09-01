# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Walk a Unitree G1 toward an axis-aligned object's nearest edge."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar

import numpy as np

from i4h_engine.status import Status
from i4h_engine.task import Task, TickContext
from i4h_tasks.basic.motion.g1_locomotion import G1_WBC_WIDTH, set_g1_wbc_command


def collision_history_key(robot: str, object_name: str) -> str:
    """Blackboard key shared by rule-based and policy success checks."""
    return f"g1_walk_to_object:{robot}:{object_name}:collided"


def edge_distance_xy(
    robot_position: np.ndarray,
    object_position: np.ndarray,
    half_extents_xy: np.ndarray | tuple[float, float],
) -> np.ndarray:
    """Horizontal distance from a point to an axis-aligned rectangle."""
    delta = np.abs(robot_position[..., :2] - object_position[..., :2])
    outside = np.maximum(delta - np.asarray(half_extents_xy, dtype=np.float32), 0.0)
    return np.linalg.norm(outside, axis=-1)


def upright_tilt_deg(quat_wxyz: np.ndarray) -> np.ndarray:
    """Angle between the robot's local and world up axes."""
    quat = np.asarray(quat_wxyz, dtype=np.float32)
    quat = quat / np.clip(np.linalg.norm(quat, axis=-1, keepdims=True), 1e-8, None)
    up_dot = 1.0 - 2.0 * (quat[..., 1] ** 2 + quat[..., 2] ** 2)
    return np.degrees(np.arccos(np.clip(up_dot, -1.0, 1.0)))


def reach_object_success(
    ctx: TickContext,
    *,
    object_name: str,
    robot: str = "robot",
    max_distance_m: float = 0.25,
    max_tilt_deg: float = 10.0,
    max_linear_speed_m_s: float = 0.05,
    max_angular_speed_rad_s: float = 0.10,
) -> np.ndarray:
    """Success mask with collision history latched in the workflow blackboard."""
    root = ctx.scene.robot_root(robot)
    target = ctx.scene.object(object_name)
    half_extents_xy = ctx.scene.footprint_half_extents(object_name)
    distance = edge_distance_xy(root.pose.pos, target.pose.pos, half_extents_xy)
    tilt = upright_tilt_deg(root.pose.quat)
    linear_speed = np.linalg.norm(root.lin_vel, axis=-1)
    angular_speed = np.linalg.norm(root.ang_vel, axis=-1)
    key = collision_history_key(robot, object_name)
    previous = np.asarray(ctx.blackboard.get(key, np.zeros(ctx.num_envs, dtype=bool)), dtype=bool)
    collided = previous | np.asarray(ctx.scene.contact(robot, object_name), dtype=bool)
    ctx.blackboard[key] = collided
    return (
        (distance > 0.0)
        & (distance <= max_distance_m)
        & (tilt <= max_tilt_deg)
        & (linear_speed <= max_linear_speed_m_s)
        & (angular_speed <= max_angular_speed_rad_s)
        & ~collided
    )


class G1WalkToObject(Task):
    """Drive G1's WBC forward and stop upright before an object's nearest edge."""

    advance_on_success = True
    requires: ClassVar[dict[str, object]] = {
        "embodiment": "g1",
        "action_space": "joint_position",
        "dof": G1_WBC_WIDTH,
        "robots": ["robot"],
    }

    @dataclass
    class Outputs:
        edge_distance_m: float = math.inf
        tilt_deg: float = math.inf
        stopped: bool = False
        collided: bool = False

    def __init__(
        self,
        *,
        object: str,
        robot: str = "robot",
        success_distance_m: float = 0.25,
        max_tilt_deg: float = 10.0,
        stop_linear_speed_m_s: float = 0.05,
        stop_angular_speed_rad_s: float = 0.10,
        stable_s: float = 0.5,
        max_forward_speed_m_s: float = 0.35,
        approach_gain: float = 0.5,
        base_height_m: float = 0.75,
        name: str | None = None,
    ) -> None:
        super().__init__(name=name)
        if success_distance_m <= 0.0:
            raise ValueError("success_distance_m must be positive")
        self.object = object
        self.robot = robot
        self.success_distance_m = success_distance_m
        self.max_tilt_deg = max_tilt_deg
        self.stop_linear_speed_m_s = stop_linear_speed_m_s
        self.stop_angular_speed_rad_s = stop_angular_speed_rad_s
        self.stable_s = stable_s
        self.max_forward_speed_m_s = max_forward_speed_m_s
        self.approach_gain = approach_gain
        self.base_height_m = base_height_m
        self._stable_ticks = 0
        self._collided: np.ndarray | None = None
        self._distance = np.array([math.inf], dtype=np.float32)
        self._tilt = np.array([math.inf], dtype=np.float32)
        self._stopped = np.array([False])
        self._approach_complete: np.ndarray | None = None

    def on_enter(self, ctx: TickContext, inputs: object) -> None:
        self._stable_ticks = 0
        self._collided = np.zeros(ctx.num_envs, dtype=bool)
        self._approach_complete = np.zeros(ctx.num_envs, dtype=bool)
        ctx.blackboard[collision_history_key(self.robot, self.object)] = self._collided.copy()

    def tick(self, ctx: TickContext) -> Status:
        root = ctx.scene.robot_root(self.robot)
        target = ctx.scene.object(self.object)
        half_extents_xy = ctx.scene.footprint_half_extents(self.object)
        self._distance = edge_distance_xy(root.pose.pos, target.pose.pos, half_extents_xy)
        self._tilt = upright_tilt_deg(root.pose.quat)
        linear_stopped = np.linalg.norm(root.lin_vel, axis=-1) <= self.stop_linear_speed_m_s
        angular_stopped = np.linalg.norm(root.ang_vel, axis=-1) <= self.stop_angular_speed_rad_s
        self._stopped = linear_stopped & angular_stopped

        assert self._collided is not None
        self._collided |= np.asarray(ctx.scene.contact(self.robot, self.object), dtype=bool)
        ctx.blackboard[collision_history_key(self.robot, self.object)] = self._collided.copy()
        if bool(self._collided.any()):
            self._command(ctx, np.zeros(ctx.num_envs, dtype=np.float32))
            return Status.FAILURE

        in_range = (self._distance > 0.0) & (self._distance <= self.success_distance_m)
        assert self._approach_complete is not None
        stop_distance = max(0.0, self.success_distance_m - 0.05)
        self._approach_complete |= (self._distance > 0.0) & (self._distance <= stop_distance + 1e-6)
        upright = self._tilt <= self.max_tilt_deg
        stable = in_range & upright & self._stopped
        if bool(stable.all()):
            self._stable_ticks += 1
        else:
            self._stable_ticks = 0

        speed = np.clip(
            self.approach_gain * self._distance,
            0.0,
            self.max_forward_speed_m_s,
        ).astype(np.float32)
        speed[self._approach_complete | stable] = 0.0
        self._command(ctx, speed)

        required_ticks = max(1, math.ceil(self.stable_s / ctx.dt))
        return Status.SUCCESS if self._stable_ticks >= required_ticks else Status.RUNNING

    def _command(self, ctx: TickContext, forward_speed: np.ndarray) -> None:
        navigation = np.column_stack([forward_speed, np.zeros(ctx.num_envs), np.zeros(ctx.num_envs)]).astype(np.float32)
        set_g1_wbc_command(
            ctx,
            navigation=navigation,
            robot=self.robot,
            base_height_m=self.base_height_m,
        )

    def on_exit(self, ctx: TickContext) -> Outputs:
        return self.Outputs(
            edge_distance_m=float(np.max(self._distance)),
            tilt_deg=float(np.max(self._tilt)),
            stopped=bool(self._stopped.all()),
            collided=bool(self._collided.any()) if self._collided is not None else False,
        )

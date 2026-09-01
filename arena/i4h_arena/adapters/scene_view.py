# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""IsaacLab scene → :class:`i4h_common.world.SceneView`.

This is the only place torch tensors become numpy. Everything above it — the
whole skill library, the engine, the workflows — is numpy and therefore testable
without a simulator. The cost is a device sync per read, which for a scripted
node at ``--envs 1`` is noise against a 60 Hz physics step; for large batches
the reads are cached per tick so a workflow with several active nodes pays once.
"""

from __future__ import annotations

import logging
from typing import Any

import numpy as np

from i4h_common.types import CameraFrame, JointState, ObjectState, Pose, quat_mul, quat_rotate
from i4h_common.world import UnsupportedActuation

logger = logging.getLogger("i4h_arena.scene")


def _np(value: Any) -> np.ndarray:
    """Detach a torch tensor to numpy, or pass numpy through."""
    if value is None:
        return np.zeros((0,), dtype=np.float32)
    detach = getattr(value, "detach", None)
    if callable(detach):
        return detach().cpu().numpy()
    numpy = getattr(value, "numpy", None)
    if callable(numpy):
        return np.asarray(numpy())
    return np.asarray(value)


def _quat_wxyz(value: Any) -> np.ndarray:
    """Convert simulator ``xyzw`` storage to common ``wxyz``."""
    quat_xyzw = _np(value)
    return quat_xyzw[..., [3, 0, 1, 2]]


class ArenaSceneView:
    """Read side of a live IsaacLab environment."""

    def __init__(
        self,
        env: Any,
        *,
        objects: tuple[str, ...] = (),
        robots: tuple[str, ...] = ("robot",),
        cameras: tuple[str, ...] = (),
        home_joints: np.ndarray | None = None,
        tcp_body: str | None = None,
        gripper: bool = True,
        object_aliases: dict[str, str] | None = None,
        footprint_half_extents: dict[str, tuple[float, float]] | None = None,
        command_objects: dict[str, tuple[str, str]] | None = None,
        robot_assets: dict[str, str] | None = None,
        joint_orders: dict[str, tuple[str, ...]] | None = None,
        tcp_sensors: dict[str, str] | None = None,
        camera_aliases: dict[str, str] | None = None,
        joint_state_providers: dict[str, Any] | None = None,
        root_relative: bool = False,
    ) -> None:
        self._env = env
        self._objects = objects
        self._robots = robots
        self._cameras = cameras
        self._tcp_body = tcp_body
        self._gripper = gripper
        self._home = home_joints
        self._object_aliases = object_aliases or {}
        self._footprint_half_extents = footprint_half_extents or {}
        self._command_objects = command_objects or {}
        self._robot_assets = robot_assets or {}
        self._joint_orders = joint_orders or {}
        self._tcp_sensors = tcp_sensors or {}
        self._camera_aliases = camera_aliases or {}
        self._joint_state_providers = joint_state_providers or {}
        self._root_relative = root_relative
        self._cache: dict[str, Any] = {}

    # -- lifecycle -------------------------------------------------------
    def invalidate(self) -> None:
        """Drop per-tick caches. The runner calls this after every ``env.step``."""
        self._cache.clear()

    @property
    def _scene(self) -> Any:
        return self._env.unwrapped.scene

    # -- SceneView -------------------------------------------------------
    @property
    def num_envs(self) -> int:
        return int(self._env.unwrapped.num_envs)

    @property
    def objects(self) -> tuple[str, ...]:
        return self._objects

    @property
    def robots(self) -> tuple[str, ...]:
        return self._robots

    def object(self, name: str) -> ObjectState:
        key = f"object:{name}"
        if key not in self._cache:
            command = self._command_objects.get(name)
            if command is not None:
                command_name, _robot = command
                value = _np(self._env.unwrapped.command_manager.get_command(command_name))
                self._cache[key] = ObjectState(
                    name=name,
                    pose=Pose(pos=value[:, :3], quat=value[:, [6, 3, 4, 5]]),
                    lin_vel=np.zeros((self.num_envs, 3), dtype=np.float32),
                    ang_vel=np.zeros((self.num_envs, 3), dtype=np.float32),
                )
                return self._cache[key]
            try:
                asset = self._scene[self._object_aliases.get(name, name)]
            except (KeyError, IndexError) as exc:
                raise KeyError(f"no scene object {name!r}; scene declares {list(self._objects)}") from exc
            data = getattr(asset, "data", None)
            if data is not None and hasattr(data, "root_pos_w"):
                root = data.root_pos_w
                quat = data.root_quat_w
                vel = getattr(data, "root_lin_vel_w", None)
                ang_vel = getattr(data, "root_ang_vel_w", None)
                pos_np = _np(root)
                quat_np = _quat_wxyz(quat)
                vel_np = _np(vel) if vel is not None else np.zeros((self.num_envs, 3), dtype=np.float32)
                ang_vel_np = _np(ang_vel) if ang_vel is not None else np.zeros((self.num_envs, 3), dtype=np.float32)
            else:
                get_world_poses = getattr(asset, "get_world_poses", None)
                if not callable(get_world_poses):
                    raise TypeError(f"scene object {name!r} exposes neither rigid-body state nor a world pose")
                root, quat = get_world_poses()
                pos_np = _np(root)[..., :3]
                quat_np = _quat_wxyz(quat)
                vel_np = np.zeros((self.num_envs, 3), dtype=np.float32)
                ang_vel_np = np.zeros((self.num_envs, 3), dtype=np.float32)
            if self._root_relative:
                pos_np, quat_np = self._to_robot_frame(pos_np, quat_np)
                root_quat_inv = self._root_pose("robot")[1].copy()
                root_quat_inv[:, 1:] *= -1.0
                vel_np = quat_rotate(root_quat_inv, vel_np)
                ang_vel_np = quat_rotate(root_quat_inv, ang_vel_np)
            self._cache[key] = ObjectState(
                name=name,
                pose=Pose(pos=pos_np, quat=quat_np),
                lin_vel=vel_np,
                ang_vel=ang_vel_np,
            )
        return self._cache[key]

    def footprint_half_extents(self, name: str) -> np.ndarray:
        try:
            values = self._footprint_half_extents[name]
        except KeyError as exc:
            raise KeyError(
                f"no collision footprint for {name!r}; scene declares {sorted(self._footprint_half_extents)}"
            ) from exc
        return np.broadcast_to(
            np.asarray(values, dtype=np.float32),
            (self.num_envs, 2),
        ).copy()

    def joints(self, robot: str = "robot") -> JointState:
        key = f"joints:{robot}"
        if key not in self._cache:
            provider = self._joint_state_providers.get(robot)
            if provider is not None:
                state = provider.joint_state()
                if not isinstance(state, JointState):
                    raise TypeError(f"joint state provider for {robot!r} must return JointState")
                self._cache[key] = state
                return state
            if robot == "robot" and robot not in self._robot_assets and len(self._robot_assets) > 1:
                parts = [self.joints(name) for name in self._robot_assets]
                self._cache[key] = JointState(
                    pos=np.concatenate([part.pos for part in parts], axis=-1),
                    vel=np.concatenate([part.vel for part in parts], axis=-1),
                    names=tuple(
                        f"{name}.{joint}"
                        for name, part in zip(self._robot_assets, parts, strict=True)
                        for joint in part.names
                    ),
                )
                return self._cache[key]
            articulation = self._scene[self._robot_assets.get(robot, robot)]
            names = tuple(articulation.data.joint_names)
            order = self._joint_orders.get(robot)
            indices = [names.index(name) for name in order] if order else slice(None)
            self._cache[key] = JointState(
                pos=_np(articulation.data.joint_pos)[:, indices],
                vel=_np(articulation.data.joint_vel)[:, indices],
                names=order or names,
            )
        return self._cache[key]

    def robot_root(self, robot: str = "robot") -> ObjectState:
        key = f"robot_root:{robot}"
        if key not in self._cache:
            articulation = self._scene[self._robot_assets.get(robot, robot)]
            self._cache[key] = ObjectState(
                name=robot,
                pose=Pose(
                    pos=_np(articulation.data.root_pos_w),
                    quat=_quat_wxyz(articulation.data.root_quat_w),
                ),
                lin_vel=_np(articulation.data.root_lin_vel_w),
                ang_vel=_np(
                    getattr(
                        articulation.data,
                        "root_ang_vel_w",
                        np.zeros((self.num_envs, 3), dtype=np.float32),
                    )
                ),
            )
        return self._cache[key]

    def home_joints(self, robot: str = "robot") -> np.ndarray:
        if self._home is not None:
            return self._home
        # Fall back to the articulation's default pose, which is what IsaacLab
        # resets to — correct for scenes that do not override a home.
        articulation = self._scene[self._robot_assets.get(robot, robot)]
        names = tuple(articulation.data.joint_names)
        order = self._joint_orders.get(robot)
        indices = [names.index(name) for name in order] if order else slice(None)
        return _np(articulation.data.default_joint_pos)[:, indices]

    def tcp(self, robot: str = "robot") -> Pose:
        key = f"tcp:{robot}"
        if key not in self._cache:
            sensor_name = self._tcp_sensors.get(robot)
            if sensor_name is not None:
                data = self._scene[sensor_name].data
                pos = _np(data.target_pos_w)[:, 0, :]
                quat = _quat_wxyz(data.target_quat_w)[:, 0, :]
                if self._root_relative:
                    pos, quat = self._to_robot_frame(pos, quat, robot)
                self._cache[key] = Pose(
                    pos=pos,
                    quat=quat,
                )
                return self._cache[key]
            articulation = self._scene[self._robot_assets.get(robot, robot)]
            body_names = list(articulation.data.body_names)
            index = (
                body_names.index(self._tcp_body)
                if self._tcp_body and self._tcp_body in body_names
                else -1  # last link is the conventional tool frame
            )
            pos = _np(articulation.data.body_pos_w[:, index])
            quat = _quat_wxyz(articulation.data.body_quat_w[:, index])
            if self._root_relative:
                pos, quat = self._to_robot_frame(pos, quat, robot)
            self._cache[key] = Pose(pos=pos, quat=quat)
        return self._cache[key]

    def _root_pose(self, robot: str) -> tuple[np.ndarray, np.ndarray]:
        articulation = self._scene[self._robot_assets.get(robot, robot)]
        return _np(articulation.data.root_pos_w), _quat_wxyz(articulation.data.root_quat_w)

    def _to_robot_frame(
        self,
        pos: np.ndarray,
        quat: np.ndarray,
        robot: str = "robot",
    ) -> tuple[np.ndarray, np.ndarray]:
        root_pos, root_quat = self._root_pose(robot)
        root_quat_inv = root_quat.copy()
        root_quat_inv[:, 1:] *= -1.0
        return (
            quat_rotate(root_quat_inv, pos - root_pos),
            quat_mul(root_quat_inv, quat),
        )

    def gripper_width(self, robot: str = "robot") -> np.ndarray:
        """Jaw opening, by the convention that the jaw is the last DOF.

        Raises on an embodiment that has no jaw — the G1 dex hands and the
        ultrasound probe would otherwise silently return an unrelated joint,
        and a grasp check reading that would be confidently wrong. workflow-lint
        normally prevents this via ``requires.gripper``.
        """
        if not self._gripper:
            raise UnsupportedActuation(f"robot {robot!r} in scene has no gripper; the scene declares gripper: false")
        return self.joints(robot).pos[:, -1]

    def contact(self, a: str, b: str) -> np.ndarray:
        """Contact between two named entities.

        Only available when the scene declares a ``ContactSensorCfg``; without
        one this reports ``False`` rather than raising, so a task's contact check
        degrades to its fallback (see ``i4h_tasks.basic.manipulation.Grasp``).
        """
        try:
            sensors = self._scene.sensors
            available = tuple(sensors)
        except (AttributeError, TypeError):
            return np.zeros(self.num_envs, dtype=bool)
        bases = (f"contact_{a}_{b}", f"contact_{b}_{a}")
        names = tuple(name for name in available if any(name == base or name.startswith(f"{base}__") for base in bases))
        if not names:
            return np.zeros(self.num_envs, dtype=bool)
        touching = np.zeros(self.num_envs, dtype=bool)
        for name in names:
            sensor = sensors[name]
            filtered_forces = _np(getattr(sensor.data, "force_matrix_w", None))
            forces = filtered_forces if filtered_forces.size else _np(sensor.data.net_forces_w)
            if forces.size:
                magnitudes = np.linalg.norm(forces, axis=-1)
                touching |= magnitudes.reshape(self.num_envs, -1).max(axis=-1) > 1e-3
        return touching

    def camera(self, name: str, *, output: str = "rgb") -> CameraFrame | None:
        if name not in self._cameras:
            return None
        key = f"camera:{name}:{output}"
        if key not in self._cache:
            try:
                sensor = self._scene[self._camera_aliases.get(name, name)]
                rgb = _np(sensor.data.output[output])
            except (KeyError, IndexError, AttributeError, TypeError):
                logger.debug("camera %s unavailable", name, exc_info=True)
                self._cache[key] = None
                return None
            frame = np.asarray(rgb)[0][..., :3].astype(np.uint8)
            self._cache[key] = CameraFrame(
                name=name,
                height=int(frame.shape[0]),
                width=int(frame.shape[1]),
                data=frame.tobytes(),
                encoding="rgb8",
            )
        return self._cache[key]

    def sensor_signal(self, name: str, output: str) -> np.ndarray | None:
        """Read a sensor output in its native dtype, before any display mapping.

        ``camera`` maps to 8-bit RGB for viewing, which folds in polarity, window and gamma.
        A recording wants the quantity the renderer produced instead, so that it stays
        reproducible no matter how the live view is set.
        """
        if name not in self._cameras:
            return None
        key = f"signal:{name}:{output}"
        if key not in self._cache:
            try:
                sensor = self._scene[self._camera_aliases.get(name, name)]
                values = _np(sensor.data.output[output])
            except (KeyError, IndexError, AttributeError, TypeError):
                logger.debug("sensor %s has no %s output", name, output, exc_info=True)
                self._cache[key] = None
                return None
            self._cache[key] = np.asarray(values)[0]
        return self._cache[key]

    def set_sensor_brightness(self, name: str, brightness: float) -> float | None:
        """Set a camera-compatible sensor's live presentation brightness."""
        sensor = self._scene[self._camera_aliases.get(name, name)]
        setter = getattr(sensor, "set_dsa_brightness", None)
        if not callable(setter):
            return None
        return float(setter(brightness))

    def set_sensor_appearance(self, name: str, appearance: str) -> str | None:
        """Select a camera-compatible sensor's display appearance, such as fluoro or xray."""
        sensor = self._scene[self._camera_aliases.get(name, name)]
        setter = getattr(sensor, "set_display_appearance", None)
        if not callable(setter):
            return None
        selected = setter(appearance)
        if selected is None:
            return None
        self.invalidate()
        return str(selected)

    def set_sensor_display_control(self, name: str, control: str, value: float) -> float | None:
        """Adjust a camera-compatible sensor's display mapping, such as the window width."""
        sensor = self._scene[self._camera_aliases.get(name, name)]
        setter = getattr(sensor, "set_display_control", None)
        if not callable(setter):
            return None
        selected = setter(control, value)
        if selected is None:
            return None
        self.invalidate()
        return float(selected)

    def recalibrate_sensor_display(self, name: str) -> bool:
        """Re-fit a camera-compatible sensor's display mapping to the next frame."""
        sensor = self._scene[self._camera_aliases.get(name, name)]
        recalibrate = getattr(sensor, "recalibrate_display", None)
        if not callable(recalibrate):
            return False
        recalibrated = bool(recalibrate())
        if recalibrated:
            self.invalidate()
        return recalibrated

    def select_sensor_projection(self, name: str, angle_rad: float) -> float | None:
        """Select a named projection on a camera-compatible medical sensor."""
        sensor = self._scene[self._camera_aliases.get(name, name)]
        select = getattr(sensor, "select_projection", None)
        if not callable(select):
            return None
        selected = float(select(angle_rad))
        self.invalidate()
        return selected

    def observation(self, group: str, name: str) -> np.ndarray:
        """Read a controller-facing observation captured by IsaacLab after reset/step."""
        key = f"observation:{group}:{name}"
        if key not in self._cache:
            obs_buf = getattr(self._env.unwrapped, "obs_buf", None)
            if not isinstance(obs_buf, dict):
                raise KeyError("environment exposes no grouped observation buffer")
            group_values = obs_buf.get(group)
            if not isinstance(group_values, dict) or name not in group_values:
                available = sorted(group_values) if isinstance(group_values, dict) else []
                raise KeyError(f"no observation {group}.{name}; available in {group}: {available}")
            self._cache[key] = _np(group_values[name])
        return self._cache[key]

    def termination(self, name: str) -> np.ndarray:
        """Return the latest value of one IsaacLab termination term.

        ``ManagerBasedRLEnv.step`` resets terminal environments before it
        returns, but ``TerminationManager.get_term`` deliberately retains the
        term values from that step.  Reading them on the next workflow tick avoids
        losing a one-step success pulse during the automatic reset.
        """
        manager = getattr(self._env.unwrapped, "termination_manager", None)
        if manager is None or name not in manager.active_terms:
            return np.zeros(self.num_envs, dtype=bool)
        return _np(manager.get_term(name)).astype(bool, copy=False)

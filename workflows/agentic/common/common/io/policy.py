# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import threading
import time
from typing import Any

import numpy as np
from common.config import get_zenoh_config
from common.io.camera import CameraSubscriber
from common.io.robot import RobotCommandPublisher, RobotStateSubscriber
from common.messages import CameraStream, RobotCommand, RobotState
from common.zenoh_utils import close_quietly, open_zenoh_session


class PolicyIOBase:
    def __init__(self, *, env_id: str, command_dtype=np.float64) -> None:
        zenoh = get_zenoh_config(env_id)
        self._session = open_zenoh_session()
        self._frames: dict[str, CameraStream] = {}
        self._state: RobotState | None = None
        self._lock = threading.Lock()
        self._camera_names = set(zenoh.camera_names)
        self._command_dtype = command_dtype
        self._camera_subscribers = [
            CameraSubscriber(
                self._session,
                lambda frame, cam_key=cam_key: self._on_camera(cam_key, frame),
                key_expr=key_expr,
            )
            for cam_key, key_expr in zenoh.camera_keys.items()
        ]
        self._state_subscriber = RobotStateSubscriber(
            self._session,
            self._on_state,
            key_expr=zenoh.robot_state_key,
        )
        self._command_publisher = RobotCommandPublisher(self._session, key_expr=zenoh.robot_command_key)

    def __enter__(self):
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        for subscriber in self._camera_subscribers:
            subscriber.close()
        self._state_subscriber.close()
        self._command_publisher.close()
        close_quietly(self._session)

    def wait_for_data(self, timeout: float = 30.0) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._camera_names.issubset(self._frames.keys()) and self._state is not None:
                    return True
            time.sleep(0.05)
        return False

    def publish_command(
        self,
        action_chunk: Any,
        *,
        dt: float,
        inference_ts: int,
        run_id: str | None = None,
        episode_index: int | None = None,
        attempt_index: int | None = None,
    ) -> None:
        chunk = action_chunk.detach().cpu().numpy() if hasattr(action_chunk, "detach") else np.asarray(action_chunk)
        chunk = np.asarray(chunk, dtype=self._command_dtype)
        if chunk.ndim == 3:
            chunk = chunk[0]
        if chunk.ndim == 1:
            chunk = chunk[np.newaxis, :]
        state = self._state
        msg = RobotCommand(
            run_id=run_id if run_id is not None else (state.run_id if state is not None else ""),
            episode_index=(
                episode_index if episode_index is not None else (state.episode_index if state is not None else 0)
            ),
            attempt_index=(
                attempt_index if attempt_index is not None else (state.attempt_index if state is not None else 0)
            ),
            horizon=int(chunk.shape[0]),
            dt=float(dt),
            inference_ts=int(inference_ts),
            joint_positions=chunk.astype(float).reshape(-1).tolist(),
        )
        self._command_publisher.publish(msg)

    def latest_observation(self) -> dict | None:
        raise NotImplementedError

    def _on_camera(self, cam_key: str, frame: CameraStream) -> None:
        with self._lock:
            self._frames[cam_key] = frame

    def _on_state(self, state: RobotState) -> None:
        with self._lock:
            self._state = state


def camera_to_array(frame: CameraStream) -> np.ndarray | None:
    if frame.height == 0 or frame.width == 0:
        return None
    image = np.frombuffer(bytes(frame.data), dtype=np.uint8)
    if image.size != frame.height * frame.width * 3:
        return None
    return image.reshape(frame.height, frame.width, 3).copy()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import threading
import time
from collections.abc import Iterator

from annotator.records import FrameBundle
from common.config import get_zenoh_config
from common.io.camera import CameraSubscriber
from common.io.policy import camera_to_array
from common.messages import CameraStream
from common.zenoh_utils import close_quietly, open_zenoh_session


class LiveCameraSampler:
    def __init__(self, *, env_id: str, cameras: list[str] | None = None) -> None:
        zenoh = get_zenoh_config(env_id)
        camera_keys = zenoh.camera_keys
        if cameras:
            missing = [camera for camera in cameras if camera not in camera_keys]
            if missing:
                raise KeyError(f"unknown cameras for {env_id}: {missing}; available: {sorted(camera_keys)}")
            camera_keys = {camera: camera_keys[camera] for camera in cameras}

        self._session = open_zenoh_session()
        self._lock = threading.Lock()
        self._frames: dict[str, CameraStream] = {}
        self._revision = 0
        self._subscribers = [
            CameraSubscriber(
                self._session,
                lambda frame, camera_name=camera_name: self._on_camera(camera_name, frame),
                key_expr=key_expr,
            )
            for camera_name, key_expr in camera_keys.items()
        ]
        self._camera_names = set(camera_keys)

    def __enter__(self) -> LiveCameraSampler:
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()

    def close(self) -> None:
        for subscriber in self._subscribers:
            subscriber.close()
        close_quietly(self._session)

    def wait_for_frames(self, timeout: float) -> bool:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._camera_names.issubset(self._frames):
                    return True
            time.sleep(0.05)
        return False

    def sample(self, *, index: int) -> FrameBundle:
        with self._lock:
            frames = dict(self._frames)
        cameras = {}
        for camera_name in sorted(self._camera_names):
            image = camera_to_array(frames[camera_name])
            if image is None:
                raise ValueError(f"camera {camera_name} did not contain an RGB frame")
            cameras[camera_name] = image
        return FrameBundle(index=index, cameras=cameras)

    def wait_for_update(self, last_revision: int, timeout: float) -> int | None:
        deadline = time.time() + timeout
        while time.time() < deadline:
            with self._lock:
                if self._revision > last_revision and self._camera_names.issubset(self._frames):
                    return self._revision
            time.sleep(0.01)
        return None

    @property
    def revision(self) -> int:
        with self._lock:
            return self._revision

    def _on_camera(self, camera_name: str, frame: CameraStream) -> None:
        with self._lock:
            self._frames[camera_name] = frame
            self._revision += 1


def iter_live_samples(
    *,
    env_id: str,
    cameras: list[str] | None,
    timeout: float,
    interval: float,
    count: int | None,
    skip_initial_frames: int = 0,
    frame_stride: int = 1,
) -> Iterator[FrameBundle]:
    frame_stride = max(1, frame_stride)
    skip_initial_frames = max(0, skip_initial_frames)
    with LiveCameraSampler(env_id=env_id, cameras=cameras) as sampler:
        if not sampler.wait_for_frames(timeout):
            raise TimeoutError(f"timed out waiting for camera frames for {env_id}")
        last_revision = sampler.revision
        for _ in range(skip_initial_frames):
            next_revision = sampler.wait_for_update(last_revision, timeout)
            if next_revision is None:
                raise TimeoutError(f"timed out waiting for camera frames for {env_id}")
            last_revision = next_revision
        index = 0
        while count is None or index < count:
            for _ in range(frame_stride):
                next_revision = sampler.wait_for_update(last_revision, timeout)
                if next_revision is None:
                    raise TimeoutError(f"timed out waiting for camera frames for {env_id}")
                last_revision = next_revision
            yield sampler.sample(index=index)
            index += 1
            if count is None or index < count:
                time.sleep(interval)

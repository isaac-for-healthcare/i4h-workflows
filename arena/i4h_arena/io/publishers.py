# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Publishing the sim's observable state onto the bus.

Two consumers today: an out-of-process teleop client that needs to see what it
is driving, and anything watching a rollout live. Remote *task* observations do
not come through here — ``RemoteTask`` publishes exactly what its manifest asks
for, on its own per-node channel, so a policy is never handed frames it did not
request.

Publishing is throttled: cameras at 60 Hz over zenoh would saturate the link and
add latency to the control loop for the benefit of nobody.
"""

from __future__ import annotations

import logging

import numpy as np

from i4h_common.bus.base import Bus
from i4h_common.bus.keys import Keys
from i4h_common.bus.messages import CameraStream, RobotState, encode
from i4h_common.world import SceneView

logger = logging.getLogger("i4h_arena.io")


class ScenePublisher:
    """Streams camera frames and robot state for one run."""

    def __init__(
        self,
        bus: Bus,
        keys: Keys,
        *,
        cameras: tuple[str, ...] = (),
        camera_every: int = 3,
        state_every: int = 1,
        run_id: str = "",
    ) -> None:
        self.bus = bus
        self.keys = keys
        self.cameras = cameras
        self.camera_every = max(1, camera_every)
        self.state_every = max(1, state_every)
        self.run_id = run_id
        self._step = 0
        self._frame_num = 0

    def publish(self, view: SceneView, *, node: str = "", episode_index: int = 0) -> None:
        """Called once per sim step by the runner."""
        if self._step % self.state_every == 0:
            self._publish_state(view, node=node, episode_index=episode_index)
        if self.cameras and self._step % self.camera_every == 0:
            self._publish_cameras(view)
            self._frame_num += 1
        self._step += 1

    def _publish_state(self, view: SceneView, *, node: str, episode_index: int) -> None:
        try:
            joints = view.joints()
            tcp = view.tcp()
        except (KeyError, AttributeError):
            logger.debug("robot state unavailable", exc_info=True)
            return
        self.bus.publish(
            self.keys.robot_state,
            encode(
                RobotState(
                    run_id=self.run_id,
                    episode_index=episode_index,
                    node=node,
                    joint_positions=[float(v) for v in np.asarray(joints.pos)[0]],
                    joint_velocities=[float(v) for v in np.asarray(joints.vel)[0]],
                    tcp_pos=[float(v) for v in np.asarray(tcp.pos)[0]],
                    tcp_quat=[float(v) for v in np.asarray(tcp.quat)[0]],
                )
            ),
        )

    def _publish_cameras(self, view: SceneView) -> None:
        for name in self.cameras:
            frame = view.camera(name)
            if frame is None:
                continue
            self.bus.publish(
                self.keys.camera(name),
                encode(
                    CameraStream(
                        name=name,
                        width=frame.width,
                        height=frame.height,
                        encoding=frame.encoding,
                        focal_len=frame.focal_len,
                        frame_num=self._frame_num,
                        data=frame.data,
                    )
                ),
            )

    def close(self) -> None:
        """Tell subscribers the stream ended, so a client stops waiting."""
        try:
            self.bus.publish(self.keys.robot_state, encode(RobotState(run_id=self.run_id, is_running=False)))
        except Exception:  # noqa: BLE001 - teardown must not raise
            logger.debug("final state publish failed", exc_info=True)

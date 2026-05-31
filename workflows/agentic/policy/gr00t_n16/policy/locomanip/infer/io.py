# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import numpy as np
import torch
from common.io.policy import PolicyIOBase
from common.messages import CameraStream


class PolicyIO(PolicyIOBase):
    def __init__(self, *, device: str, env_id: str, num_envs: int = 1) -> None:
        super().__init__(env_id=env_id, command_dtype=np.float32)
        self._device = device
        self._num_envs = num_envs

    def latest_observation(self) -> Optional[dict]:
        with self._lock:
            if self._state is None or "head" not in self._frames:
                return None
            head = _camera_to_tensor(self._frames["head"], self._device, self._num_envs)
            joints = torch.as_tensor(self._state.joint_positions, device=self._device, dtype=torch.float32).reshape(
                self._num_envs, -1
            )
            camera_obs: dict[str, torch.Tensor] = {"robot_head_cam_rgb": head}
            for cam_label, frame in self._frames.items():
                if cam_label == "head":
                    continue
                camera_obs[f"robot_{cam_label}_cam_rgb"] = _camera_to_tensor(frame, self._device, self._num_envs)
            return {
                "camera_obs": camera_obs,
                "policy": {"robot_joint_pos": joints},
                "state_ts": self._state.ts,
                "run_id": self._state.run_id,
                "episode_index": self._state.episode_index,
                "attempt_index": self._state.attempt_index,
            }


def _camera_to_tensor(frame: CameraStream, device: str, num_envs: int) -> torch.Tensor:
    image = np.frombuffer(bytes(frame.data), dtype=np.uint8).reshape(frame.height, frame.width, 3).copy()
    tensor = torch.as_tensor(image, device=device)
    return tensor.unsqueeze(0).repeat(num_envs, 1, 1, 1)

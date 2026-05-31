# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Zenoh IO for the Franka ultrasound policy.

Subscribes to the ultrasound room/wrist cameras + robot state and publishes
joint-position commands back to Arena.
"""

from __future__ import annotations

from typing import Optional

import numpy as np
from common.io.policy import PolicyIOBase, camera_to_array


class PolicyIO(PolicyIOBase):
    def latest_observation(self) -> Optional[dict]:
        with self._lock:
            if self._state is None or not self._camera_names.issubset(self._frames):
                return None
            obs: dict = {}
            for cam_key, frame in self._frames.items():
                image = camera_to_array(frame)
                if image is None:
                    return None
                obs[cam_key] = image
            obs["joint_positions"] = np.asarray(self._state.joint_positions, dtype=np.float64).copy()
            obs["state_ts"] = self._state.ts
            obs["run_id"] = self._state.run_id
            obs["episode_index"] = self._state.episode_index
            obs["attempt_index"] = self._state.attempt_index
        return obs

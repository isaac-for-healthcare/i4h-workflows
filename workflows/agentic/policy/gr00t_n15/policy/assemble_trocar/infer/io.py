# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from typing import Optional

import numpy as np
from common.io.policy import PolicyIOBase, camera_to_array


class PolicyIO(PolicyIOBase):
    def __init__(self, *, env_id: str) -> None:
        super().__init__(env_id=env_id, command_dtype=np.float32)

    def latest_observation(self) -> Optional[dict]:
        with self._lock:
            if self._state is None or not self._camera_names.issubset(self._frames):
                return None
            obs = {
                "frames": {key: camera_to_array(frame) for key, frame in self._frames.items()},
                "joint_positions": np.asarray(self._state.joint_positions, dtype=np.float32).copy(),
                "state_ts": self._state.ts,
                "run_id": self._state.run_id,
                "episode_index": self._state.episode_index,
                "attempt_index": self._state.attempt_index,
            }
        if any(frame is None for frame in obs["frames"].values()):
            return None
        return obs

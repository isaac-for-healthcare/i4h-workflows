# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The liver-scan PI0 checkpoint, served by openpi's websocket server, as an Arena policy.

Arena's openpi client replays chunks from any openpi-protocol server through an
embodiment adapter; this is the adapter for the Franka ultrasound probe. Start the
server first::

    tasks/openpi_pi0/.venv/bin/python -m i4h_tasks.openpi_pi0.serve ultrasound_liver_scan --port 8000
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg
from isaaclab_arena.policy.remote_policy_base import EmbodimentAdapter, RemoteChunkReplayPolicy
from openpi_client import image_tools


class FrankaProbeAdapter(EmbodimentAdapter):
    """The liver-scan checkpoint's request: seven arm joint angles, room and wrist views at 224 px."""

    #: Relative probe pose command: position delta and rotation vector.
    action_dim = 6
    image_size = (224, 224)
    arm_joints = 7

    def extract(self, observation: dict[str, Any], env_id: int) -> dict[str, np.ndarray]:
        policy = observation["policy"]
        return {
            "state": policy["joint_pos"][env_id, : self.arm_joints].detach().cpu().numpy().astype(np.float64),
            "room": policy["room"][env_id].detach().cpu().numpy(),
            "wrist": policy["wrist"][env_id].detach().cpu().numpy(),
        }

    def pack_request(self, extracted: dict[str, np.ndarray], language_instruction: str) -> dict[str, Any]:
        return {
            "observation/state": extracted["state"],
            "observation/image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(extracted["room"], *self.image_size)
            ),
            "observation/wrist_image": image_tools.convert_to_uint8(
                image_tools.resize_with_pad(extracted["wrist"], *self.image_size)
            ),
            "prompt": language_instruction,
        }


@dataclass
class LiverScanPi0PolicyCfg(PolicyCfg):
    remote_host: str = "localhost"
    remote_port: int = 8000
    open_loop_horizon: int = 5
    """Actions executed from each 50-step chunk before the next request."""

    policy_device: str = "cuda"
    ping_interval: float | None = 20.0
    ping_timeout: float | None = 300.0


@register_policy
class LiverScanPi0Policy(RemoteChunkReplayPolicy, PolicyBase[LiverScanPi0PolicyCfg]):
    """Closed-loop PI0 liver scan with chunk replay, one request per env."""

    name = "i4h_liver_scan_pi0"
    server_response_actions_key = "actions"

    def __init__(self, config: LiverScanPi0PolicyCfg) -> None:
        super().__init__(config, FrankaProbeAdapter(), config.open_loop_horizon)

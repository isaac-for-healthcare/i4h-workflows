# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A GR00T N1.7 checkpoint served by GR00T's inference server, as an Arena policy.

GR00T 1.7's server speaks the protocol of GR00T's ``PolicyClient``: nested
``video`` / ``state`` / ``language`` requests, answered with one chunk per action
group. Arena's GR00T client builds those requests for Arena's own embodiments
from its joint-space configs; this one builds them for the i4h checkpoints from the
env's observations and replays the first ``open_loop_horizon`` actions of each
chunk. The language key is the one the served checkpoint declares. Start the
server first::

    tasks/gr00t_n17/.venv/bin/python -m i4h_tasks.gr00t_n17.serve scissor_pick_and_place --port 5555
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
from gr00t.policy.server_client import PolicyClient
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_arena.medical.catheter_recording import downsample_frame
from i4h_arena.policies._chunks import ChunkReplay, from_lerobot, to_lerobot, uint8_image

#: Per checkpoint: the observation group, the terms concatenated into its state vector,
#: (camera term -> video key), (state group -> slice of that vector), the action groups
#: in action order, how many actions of a chunk to execute, the robot whose LeRobot
#: joint calibration the model works in, if any, and the edge its training frames were
#: box-averaged to when recorded, if any.
LAYOUTS = {
    "so101_scissors": {
        "group": "policy",
        "state_terms": ("joint_pos",),
        "video": {"room": "room", "wrist": "wrist"},
        "state": {"single_arm": slice(0, 5), "gripper": slice(5, 6)},
        "action": ("single_arm", "gripper"),
        "open_loop_horizon": 16,
        "lerobot_units": "so101",
        "frame_edge": None,
    },
    # The endoluminal catheter, armless or arm-borne: the drive's insertion, twist and tip
    # bend and the C-arm angle, then the target offset and route state its patient twin
    # supplies. The model commands the same four rates.
    "catheter": {
        "group": "policy",
        "state_terms": ("drive_state", "target_offset", "route_state"),
        "video": {"fluoroscopy_rgb": "fluoroscopy"},
        "state": {"catheter": slice(0, 3), "carm": slice(3, 4), "target": slice(4, 7), "route": slice(7, 9)},
        "action": ("catheter", "carm"),
        "open_loop_horizon": 16,
        "lerobot_units": None,
        # The detector renders 1024 px; recordings, and so the training data, hold 256.
        "frame_edge": 256,
    },
}


@dataclass
class Gr00tN17RemotePolicyCfg(PolicyCfg):
    remote_host: str = "localhost"
    remote_port: int = 5555
    embodiment: str = "so101_scissors"
    """Observation and action layout of the checkpoint: one of LAYOUTS."""
    open_loop_horizon: int = 0
    """Actions executed from each inferred chunk before the next request; 0 uses the layout's."""


@register_policy
class Gr00tN17RemotePolicy(ChunkReplay, PolicyBase[Gr00tN17RemotePolicyCfg]):
    """Closed-loop GR00T N1.7 inference with chunk replay, one request per env."""

    name = "i4h_gr00t_n17_remote"

    def __init__(self, config: Gr00tN17RemotePolicyCfg):
        super().__init__(config)
        self.layout = LAYOUTS[config.embodiment]
        self.horizon = config.open_loop_horizon or self.layout["open_loop_horizon"]
        self._client = PolicyClient(host=config.remote_host, port=config.remote_port)
        self._language_key: str | None = None
        self.task_description: str | None = None

    def _language(self) -> str:
        if self._language_key is None:
            self._language_key = self._client.get_modality_config()["language"].modality_keys[0]
        return self._language_key

    def _infer(self, observation: dict, env_id: int, action_dim: int) -> torch.Tensor:
        layout = self.layout
        group = observation[layout["group"]]
        joints = torch.cat([group[term][env_id].double().cpu() for term in layout["state_terms"]]).numpy()
        if layout["lerobot_units"]:
            joints = to_lerobot(joints, layout["lerobot_units"])
        joints = joints.astype(np.float32)
        request = {
            "video": {
                key: self._frame(group[camera][env_id])[np.newaxis, np.newaxis]
                for camera, key in layout["video"].items()
            },
            "state": {key: joints[np.newaxis, np.newaxis, part] for key, part in layout["state"].items()},
            "language": {self._language(): [[self.task_description]]},
        }
        actions, _info = self._client.get_action(request)
        # One (1, horizon, width) or (horizon,) chunk per action group, in action order.
        parts = [np.asarray(_group(actions, key), dtype=np.float64) for key in layout["action"]]
        chunk = np.concatenate([part.reshape(-1, part.shape[-1] if part.ndim > 1 else 1) for part in parts], axis=-1)
        if layout["lerobot_units"]:
            chunk = from_lerobot(chunk, layout["lerobot_units"])
        full = torch.zeros(chunk.shape[0], action_dim)
        full[:, : chunk.shape[-1]] = torch.from_numpy(chunk).float()
        return full

    def _frame(self, image: torch.Tensor) -> np.ndarray:
        frame = uint8_image(image.cpu()).numpy()
        edge = self.layout["frame_edge"]
        return downsample_frame(frame, edge) if edge else frame


def _group(actions: dict, key: str) -> np.ndarray:
    for candidate in (key, f"action.{key}"):
        if candidate in actions:
            return actions[candidate]
    raise KeyError(f"no action group {key!r} in the reply; got {sorted(actions)}")

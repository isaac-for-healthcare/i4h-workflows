# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A GR00T N1.5 checkpoint served by GR00T 1.5's ``RobotInferenceServer``, as an Arena policy.

Arena's GR00T client speaks the N1.6 protocol. N1.5 serves ``torch.save`` messages over
ZMQ request-reply with flat ``video.*`` / ``state.*`` keys, so this client builds those
from the env's observations, as Arena's remote policies build theirs, and replays the
first ``open_loop_horizon`` actions of each chunk. Start the server first::

    tasks/gr00t_n15/.venv/bin/python -m i4h_tasks.gr00t_n15.serve assemble_trocar --port 5555
"""

from __future__ import annotations

import io
from dataclasses import dataclass

import torch
import zmq
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_arena.policies._chunks import ChunkReplay, from_lerobot, to_lerobot, uint8_image

#: Per checkpoint: the observation group, its joint term, (camera term -> video key),
#: (state key -> slice of the joints), the action keys in action order, how many leading
#: action columns the model leaves at zero, how many actions of a chunk to execute, whether
#: requests carry a batch axis ahead of the time axis, and the robot whose LeRobot joint
#: calibration the model works in, if any.
LAYOUTS = {
    "g1_trocar": {
        "group": "policy",
        "joints": "assemble_trocar_joint_pos",
        "video": {
            "front_camera_rgb": "video.room_view",
            "left_wrist_camera_rgb": "video.left_wrist_view",
            "right_wrist_camera_rgb": "video.right_wrist_view",
        },
        "state": {
            "state.left_arm": slice(15, 22),
            "state.right_arm": slice(22, 29),
            "state.left_hand": slice(29, 36),
            "state.right_hand": slice(36, 43),
        },
        "action": ("action.left_arm", "action.right_arm", "action.left_hand", "action.right_hand"),
        # The checkpoint drives arms and hands; the 15 leg and waist targets stay at zero.
        "action_offset": 15,
        "open_loop_horizon": 1,
        "batched": True,
        "lerobot_units": None,
    },
    "so101_scissors": {
        "group": "policy",
        "joints": "joint_pos",
        "video": {"room": "video.room", "wrist": "video.wrist"},
        "state": {"state.single_arm": slice(0, 5), "state.gripper": slice(5, 6)},
        "action": ("action.single_arm", "action.gripper"),
        "action_offset": 0,
        "open_loop_horizon": 16,
        "batched": False,
        # Trained on LeRobot recordings: joints in the arm's calibrated LeRobot range.
        "lerobot_units": "so101",
    },
}


@dataclass
class Gr00tN15RemotePolicyCfg(PolicyCfg):
    remote_host: str = "localhost"
    remote_port: int = 5555
    embodiment: str = "g1_trocar"
    """Observation and action layout of the checkpoint: one of LAYOUTS."""
    open_loop_horizon: int = 0
    """Actions executed from each inferred chunk before the next request; 0 uses the layout's."""
    timeout_s: float = 120.0
    """How long to wait for one inference; the first one also warms the model up."""


@register_policy
class Gr00tN15RemotePolicy(ChunkReplay, PolicyBase[Gr00tN15RemotePolicyCfg]):
    """Closed-loop GR00T N1.5 inference with chunk replay, one request per env."""

    name = "i4h_gr00t_n15_remote"

    def __init__(self, config: Gr00tN15RemotePolicyCfg):
        super().__init__(config)
        self.layout = LAYOUTS[config.embodiment]
        self.horizon = config.open_loop_horizon or self.layout["open_loop_horizon"]
        self._context = zmq.Context()
        self._socket = self._context.socket(zmq.REQ)
        self._socket.setsockopt(zmq.RCVTIMEO, int(config.timeout_s * 1000))
        self._socket.connect(f"tcp://{config.remote_host}:{config.remote_port}")
        self.task_description: str | None = None

    def _call(self, observations: dict) -> dict:
        buffer = io.BytesIO()
        torch.save({"endpoint": "get_action", "data": observations}, buffer)
        self._socket.send(buffer.getvalue())
        reply = self._socket.recv()
        if reply == b"ERROR":
            raise RuntimeError("GR00T N1.5 server failed to infer; see its log")
        return torch.load(io.BytesIO(reply), weights_only=False)

    def _infer(self, observation: dict, env_id: int, action_dim: int) -> torch.Tensor:
        layout = self.layout
        lead = (1, 1) if layout["batched"] else (1,)
        group = observation[layout["group"]]
        joints = group[layout["joints"]][env_id].float().cpu()
        if layout["lerobot_units"]:
            joints = torch.from_numpy(to_lerobot(joints.double().numpy(), layout["lerobot_units"]))
        request = {key: joints[part].reshape(*lead, -1) for key, part in layout["state"].items()}
        for camera, key in layout["video"].items():
            image = uint8_image(group[camera][env_id].cpu())
            request[key] = image.reshape(*lead, *image.shape)
        prompt = self.task_description
        request["annotation.human.task_description"] = [prompt] if layout["batched"] else prompt
        actions = self._call(request)
        # One chunk per modality in action order: (1, horizon, width) for a batch of one,
        # (horizon, width) unbatched, or (horizon,) for a single value.
        parts = [torch.as_tensor(actions[key]).double() for key in layout["action"]]
        chunk = torch.cat([part.reshape(-1, part.shape[-1] if part.ndim > 1 else 1) for part in parts], dim=-1)
        if layout["lerobot_units"]:
            chunk = torch.from_numpy(from_lerobot(chunk.numpy(), layout["lerobot_units"]))
        chunk = chunk.float()
        full = torch.zeros(chunk.shape[0], action_dim)
        full[:, layout["action_offset"] : layout["action_offset"] + chunk.shape[-1]] = chunk
        return full

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

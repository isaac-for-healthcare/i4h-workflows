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

import gymnasium as gym
import torch
import zmq
from isaaclab_arena.assets.register import register_policy
from isaaclab_arena.policy.policy_base import PolicyBase, PolicyCfg

from i4h_common.config import get_robot_config
from i4h_common.joint_utils import isaaclab_rad_to_lerobot, lerobot_to_isaaclab_rad

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
class Gr00tN15RemotePolicy(PolicyBase[Gr00tN15RemotePolicyCfg]):
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
        self._chunk: torch.Tensor | None = None
        self._index: torch.Tensor | None = None
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
            robot = get_robot_config(layout["lerobot_units"])
            joints = torch.from_numpy(
                isaaclab_rad_to_lerobot(
                    joints.double().numpy(), robot.isaaclab_joint_pos_limit_range, robot.lerobot_joint_pos_limit_range
                )
            )
        request = {key: joints[part].reshape(*lead, -1) for key, part in layout["state"].items()}
        for camera, key in layout["video"].items():
            image = group[camera][env_id].cpu()
            if image.is_floating_point():
                image = (image * 255.0 if image.max() <= 1.0 else image).clamp(0, 255)
            request[key] = image.to(torch.uint8).reshape(*lead, *image.shape)
        prompt = self.task_description
        request["annotation.human.task_description"] = [prompt] if layout["batched"] else prompt
        actions = self._call(request)
        # One chunk per modality in action order: (1, horizon, width) for a batch of one,
        # (horizon, width) unbatched, or (horizon,) for a single value.
        parts = [torch.as_tensor(actions[key]).double() for key in layout["action"]]
        chunk = torch.cat([part.reshape(-1, part.shape[-1] if part.ndim > 1 else 1) for part in parts], dim=-1)
        if layout["lerobot_units"]:
            chunk = torch.from_numpy(
                lerobot_to_isaaclab_rad(
                    chunk.numpy(), robot.lerobot_joint_pos_limit_range, robot.isaaclab_joint_pos_limit_range
                )
            )
        chunk = chunk.float()
        full = torch.zeros(chunk.shape[0], action_dim)
        full[:, layout["action_offset"] : layout["action_offset"] + chunk.shape[-1]] = chunk
        return full

    def get_action(self, env: gym.Env, observation) -> torch.Tensor:
        env = env.unwrapped
        n, action_dim = env.num_envs, env.action_manager.total_action_dim
        if self._chunk is None:
            self._chunk = torch.zeros(n, self.horizon, action_dim)
            self._index = torch.full((n,), self.horizon, dtype=torch.long)
        for env_id in range(n):
            if self._index[env_id] >= self.horizon:
                self._chunk[env_id] = self._infer(observation, env_id, action_dim)[: self.horizon]
                self._index[env_id] = 0
        action = self._chunk[torch.arange(n), self._index]
        self._index += 1
        return action.to(env.device)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._index is not None:
            self._index[slice(None) if env_ids is None else env_ids.cpu()] = self.horizon

    def close(self) -> None:
        self._socket.close(linger=0)
        self._context.term()

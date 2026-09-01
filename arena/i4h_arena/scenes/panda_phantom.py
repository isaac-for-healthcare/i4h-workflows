# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka Panda with an ultrasound probe over an abdominal phantom.

The only ee_pose scene with a Vention table rather than Props/Table, and the
only one whose default backend is openpi PI0 — neither fact appears anywhere in
this class, because the backend is a manifest lookup.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np

from i4h_arena.adapters.actuation import RobotSlice
from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config


class PandaPhantomScene(Scene):
    name = "panda_phantom"

    def register_assets(self) -> None:
        import i4h_arena.assets.panda_phantom  # noqa: F401,PLC0415

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415

        from i4h_arena.assets.panda_phantom import make_assets  # noqa: PLC0415
        from i4h_arena.embodiments.franka import FrankaUltrasoundEmbodiment  # noqa: PLC0415
        from i4h_arena.envcfg.panda_phantom import PandaPhantomEnvCfg  # noqa: PLC0415

        steps = self.args.episode_steps or self.spec.max_steps
        embodiment = FrankaUltrasoundEmbodiment(enable_cameras=bool(getattr(self.args, "enable_cameras", True)))

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=ArenaScene(assets=make_assets()),
            # PandaPhantomEnvCfg takes no env_spacing: the phantom scene is a
            # single workspace, not a tiled grid.
            task=PandaPhantomEnvCfg(episode_length_s=max(10.0, (steps + 1) / self.spec.control_hz)),
        )

    def home_joints(self, env: Any) -> np.ndarray | None:
        home = get_robot_config(self.spec.embodiment).home_joint_pos_rad
        if not home:
            return None
        return np.tile(np.asarray(home, dtype=np.float32), (int(env.unwrapped.num_envs), 1))

    def tcp_body(self) -> str | None:
        return "TCP"

    def tcp_sensors(self) -> dict[str, str]:
        return {"robot": "ee_frame"}

    def camera_aliases(self) -> dict[str, str]:
        return {"room": "room_camera", "wrist": "wrist_camera"}

    def relative_ee(self) -> bool:
        # The Panda uses a 6-D relative Cartesian controller. ArenaActuation
        # converts scripted absolute targets and passes policy deltas directly.
        return True

    def robot_slices(self, env: Any) -> tuple[RobotSlice, ...]:
        # Probe, not a gripper: no jaw column to claim.
        width = int(env.action_space.shape[-1])
        return (RobotSlice("robot", 0, width, gripper_index=None),)

    def on_reset(self, env: Any, view: Any) -> None:
        """Servo to the PI0 training start pose before publishing frame zero."""
        import torch  # noqa: PLC0415
        from isaaclab.utils.math import compute_pose_error  # noqa: PLC0415
        from isaaclab.utils.math import quat_from_euler_xyz

        from i4h_arena.envcfg.panda_phantom import reset_ultrasound_success_state  # noqa: PLC0415
        from i4h_arena.tensor_utils import to_torch  # noqa: PLC0415

        unwrapped = env.unwrapped
        device = unwrapped.device
        num_envs = unwrapped.num_envs
        setup_pos = torch.tensor((0.3229, -0.0110, 0.3000), device=device, dtype=torch.float32)
        down_quat = quat_from_euler_xyz(
            torch.tensor(math.pi, device=device),
            torch.tensor(0.0, device=device),
            torch.tensor(math.pi, device=device),
        )
        target_pos = setup_pos.repeat(num_envs, 1)
        target_quat = down_quat.repeat(num_envs, 1)

        for _ in range(40):
            ee_data = unwrapped.scene["ee_frame"].data
            current_pos = to_torch(ee_data.target_pos_w)[:, 0, :] - unwrapped.scene.env_origins
            current_quat = to_torch(ee_data.target_quat_w)[:, 0, :]
            delta_pos, delta_angle = compute_pose_error(
                current_pos,
                current_quat,
                target_pos,
                target_quat,
                rot_error_type="axis_angle",
            )
            env.step(torch.cat([delta_pos, delta_angle], dim=-1))

        env_ids = torch.arange(num_envs, device=device)
        reset_ultrasound_success_state(unwrapped, env_ids)
        view.invalidate()

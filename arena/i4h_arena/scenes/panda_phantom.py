# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka Panda with an ultrasound probe over an abdominal phantom.

The only ee_pose scene with a Vention table rather than Props/Table, and the
only one whose default backend is openpi PI0 — neither fact appears anywhere in
this class, because the backend is a manifest lookup.
"""

from __future__ import annotations

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

    def configure_env_cfg(self, env_cfg: Any) -> None:
        from isaaclab.devices import Se3KeyboardCfg  # noqa: PLC0415
        from isaaclab.devices.device_base import DevicesCfg  # noqa: PLC0415

        # Isaac Lab's record_demos.py --teleop_device keyboard finds the device here. Its six
        # values are the relative pose command; the probe has no gripper.
        env_cfg.teleop_devices = DevicesCfg(
            devices={
                "keyboard": Se3KeyboardCfg(
                    pos_sensitivity=0.05, rot_sensitivity=0.15, gripper_term=False, sim_device=env_cfg.sim.device
                )
            }
        )

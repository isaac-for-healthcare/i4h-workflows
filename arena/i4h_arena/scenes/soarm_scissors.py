# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""SO-ARM 101 over a table with surgical scissors and a tray.

Named for the robot and what is on the surface. The surface word is dropped
because 8 of 11 scenes sit on the same ``Props/Table/table.usd``; the
*instruments* are surgical, the table asset is the generic one.
"""

from __future__ import annotations

import argparse
from typing import Any

import numpy as np

from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config


class SoArmScissorsScene(Scene):
    name = "soarm_scissors"

    def register_assets(self) -> None:
        # Import for the @register_asset side effects only.
        import i4h_arena.assets.soarm_scissors  # noqa: F401,PLC0415

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415

        from i4h_arena.assets.soarm_scissors import make_assets  # noqa: PLC0415
        from i4h_arena.embodiments.so_arm import SoArm101Embodiment  # noqa: PLC0415

        print("[arena] embodiment", flush=True)
        robot = get_robot_config(self.spec.embodiment)
        embodiment = SoArm101Embodiment(
            enable_cameras=bool(getattr(self.args, "enable_cameras", True)),
            home_joint_pos_rad=robot.home_joint_pos_rad,
        )
        print("[arena] make_assets", flush=True)
        assets = make_assets()
        print("[arena] task", flush=True)
        task = self._task()
        print("[arena] IsaacLabArenaEnvironment", flush=True)
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=ArenaScene(assets=assets),
            task=task,
        )

    def _task(self) -> Any:
        """A minimal IsaacLab task; the workflow owns the success goal."""
        print("[arena] importing envcfg", flush=True)
        from i4h_arena.envcfg.soarm_scissors import SoArmScissorsEnvCfg  # noqa: PLC0415

        print("[arena] envcfg imported", flush=True)
        steps = self.args.episode_steps or self.spec.max_steps
        return SoArmScissorsEnvCfg(
            episode_length_s=max(8.0, (steps + 1) / self.spec.control_hz),
            env_spacing=float(getattr(self.args, "env_spacing", 4.0)),
            home_joint_pos_rad=get_robot_config(self.spec.embodiment).home_joint_pos_rad,
        )

    # -- adapters --------------------------------------------------------
    def home_joints(self, env: Any) -> np.ndarray:
        home = np.asarray(get_robot_config(self.spec.embodiment).home_joint_pos_rad, dtype=np.float32)
        return np.tile(home, (int(env.unwrapped.num_envs), 1))

    def tcp_body(self) -> str | None:
        return "gripper"

    def configure_args(self, args: argparse.Namespace) -> None:
        super().configure_args(args)
        # The SO-ARM has no IK; nothing in this scene can accept a Cartesian
        # target, which the scene manifest declares as joint_position so
        # workflow-lint rejects tasks/ik nodes before Isaac boots.

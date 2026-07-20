# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging

from arena.environments.core.humanoid_base import HumanoidEnvironmentBase
from common.config import get_robot_config

logger = logging.getLogger("arena")
_G1_CONFIG = get_robot_config("g1")


class AssembleTrocarEnvironment(HumanoidEnvironmentBase):
    name: str = "assemble_trocar"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        # Override the base default so --help shows the dex-3 embodiment.
        for action in parser._actions:
            if action.dest == "embodiment":
                action.default = _G1_CONFIG.assemble_trocar_embodiment_name or "g1_assemble_trocar_joint"
                break

    def configure_args(self, args: argparse.Namespace) -> None:
        # Trocar reproducibility relies on a fixed seed (matches the upstream
        # rheo workflow). Apply before AppLauncher consumes ``args.seed``.
        if args.seed is None:
            args.seed = 4
        super().configure_args(args)

    def get_env(self, args: argparse.Namespace):
        from arena.tasks.assemble_trocar import AssembleTrocarTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("trocar_assembly_scene")()
        trocar_1 = self.asset_registry.get_asset_by_name("trocar_1")()
        trocar_2 = self.asset_registry.get_asset_by_name("trocar_2")()
        tray = self.asset_registry.get_asset_by_name("tray")()
        embodiment = self.asset_registry.get_asset_by_name(args.embodiment)(enable_cameras=args.enable_cameras)

        trocar_1.set_initial_pose(
            Pose(position_xyz=(-1.60202, 1.91362, 0.87183), rotation_wxyz=(0.0, -0.0, 0.70711, 0.70711))
        )
        trocar_2.set_initial_pose(
            Pose(position_xyz=(-1.50635, 1.90997, 0.8631), rotation_wxyz=(0.69692, -0.71475, -0.000243, 0.05853))
        )
        tray.set_initial_pose(
            Pose(position_xyz=(-1.54919, 2.03365, 0.84554), rotation_wxyz=(0.70711, 0.0, 0.0, -0.70711))
        )

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background, trocar_1, trocar_2, tray]),
            task=AssembleTrocarTask(episode_length_s=20.0),
            teleop_device=None,
        )

    def run(self, args, env, app, controller) -> None:
        if not args.headless:
            _set_viewport_camera("/World/envs/env_0/Robot/d435_link/front_cam")
        super().run(args, env, app, controller)


def _set_viewport_camera(camera_prim_path: str) -> None:
    try:
        import omni.kit.commands
        from omni.kit.viewport.utility import get_active_viewport
    except Exception as exc:
        logger.debug("could not import viewport camera utilities: %s", exc)
        return

    viewport_api = get_active_viewport()
    if viewport_api is None:
        logger.debug("no active viewport found; leaving camera unchanged")
        return
    omni.kit.commands.execute("SetViewportCamera", camera_path=camera_prim_path, viewport_api=viewport_api)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from arena.environments.core.humanoid_base import HumanoidEnvironmentBase


class LocomanipTrayPickAndPlaceEnvironment(HumanoidEnvironmentBase):
    name: str = "locomanip_tray_pick_and_place"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        parser.add_argument("--object", type=str, default="surgical_tray")

    def get_env(self, args: argparse.Namespace):
        from arena.tasks.tray_pick_and_place import TrayPickPlaceTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("pre_op")()
        pick_up_object = self.asset_registry.get_asset_by_name(args.object)()
        destination_cart = self.asset_registry.get_asset_by_name("cart")()
        embodiment = self.asset_registry.get_asset_by_name(args.embodiment)(enable_cameras=args.enable_cameras)

        background.set_initial_pose(Pose(position_xyz=(4.0, 0.0, -0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        pick_up_object.set_initial_pose(Pose(position_xyz=(-1.15, -1.6, -0.08), rotation_wxyz=(0.707, 0.0, 0.0, 0.707)))
        destination_cart.set_initial_pose(Pose(position_xyz=(0.35, -1.65, -0.7875), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        embodiment.set_initial_pose(Pose(position_xyz=(-0.5, -1.62, 0.0), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

        self._maybe_patch_locomanip_mimic(args, embodiment)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background, pick_up_object, destination_cart]),
            task=TrayPickPlaceTask(pick_up_object, destination_cart, background, episode_length_s=30.0),
            teleop_device=self._resolve_teleop_device(args.teleop_device),
        )

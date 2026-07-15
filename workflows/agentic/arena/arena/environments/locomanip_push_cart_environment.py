# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse

from arena.environments.core.humanoid_base import HumanoidEnvironmentBase


class LocomanipPushCartEnvironment(HumanoidEnvironmentBase):
    name: str = "locomanip_push_cart"

    @classmethod
    def add_cli_args(cls, parser: argparse.ArgumentParser) -> None:
        super().add_cli_args(parser)
        parser.add_argument("--object", type=str, default="cart")

    def get_env(self, args: argparse.Namespace):
        from arena.tasks.push_cart import PushCartTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose

        background = self.asset_registry.get_asset_by_name("pre_op")()
        pick_up_object = self.asset_registry.get_asset_by_name("surgical_tray")()
        destination_cart = self.asset_registry.get_asset_by_name(args.object)()
        embodiment = self.asset_registry.get_asset_by_name(args.embodiment)(enable_cameras=args.enable_cameras)

        background.set_initial_pose(Pose(position_xyz=(4.0, 0.0, -0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        pick_up_object.set_initial_pose(Pose(position_xyz=(0.35, -1.65, 0.10), rotation_wxyz=(0.707, 0.0, 0.0, 0.707)))
        destination_cart.set_initial_pose(Pose(position_xyz=(0.35, -1.65, -0.7875), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        embodiment.set_initial_pose(Pose(position_xyz=(-0.4, -1.62, 0.0), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))

        self._maybe_patch_locomanip_mimic(args, embodiment)

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=Scene(assets=[background, pick_up_object, destination_cart]),
            task=PushCartTask(pick_up_object, destination_cart, background, episode_length_s=40.0),
            teleop_device=self._resolve_teleop_device(args.teleop_device),
        )

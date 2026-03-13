# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import argparse

from isaaclab_arena_environments.example_environment_base import ExampleEnvironmentBase


class G1LocomanipTrayPickAndPlaceEnvironment(ExampleEnvironmentBase):
    name: str = "g1_locomanip_tray_pick_and_place"

    def get_env(self, args_cli: argparse.Namespace):
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene
        from isaaclab_arena.utils.pose import Pose
        from simulation.tasks.g1_tray_pick_and_place_task import G1TrayPickPlaceTask

        background = self.asset_registry.get_asset_by_name("pre_op")()
        pick_up_object = self.asset_registry.get_asset_by_name(args_cli.object)()
        destination_cart = self.asset_registry.get_asset_by_name("cart")()
        embodiment = self.asset_registry.get_asset_by_name(args_cli.embodiment)(enable_cameras=args_cli.enable_cameras)

        if args_cli.teleop_device is not None:
            teleop_device = self.device_registry.get_device_by_name(args_cli.teleop_device)()
        else:
            teleop_device = None

        background.set_initial_pose(Pose(position_xyz=(4.0, 0.0, -0.8), rotation_wxyz=(1.0, 0.0, 0.0, 0.0)))
        pick_up_object.set_initial_pose(
            Pose(
                position_xyz=(-1.15, -1.6, -0.08),
                rotation_wxyz=(0.707, 0.0, 0.0, 0.707),  # Rotate 90° around Z-axis
            )
        )
        destination_cart.set_initial_pose(
            Pose(
                position_xyz=(0.35, -1.65, -0.7875),
                rotation_wxyz=(1.0, 0.0, 0.0, 0.0),
            )
        )
        embodiment.set_initial_pose(Pose(position_xyz=(-0.5, -1.62, 0.0), rotation_wxyz=(0.0, 0.0, 0.0, 1.0)))

        if (
            args_cli.embodiment == "g1_wbc_pink"
            and hasattr(args_cli, "mimic")
            and args_cli.mimic
            and not hasattr(args_cli, "auto")
        ):
            # Patch the Mimic generate function for locomanip use case
            from isaaclab_arena.utils.locomanip_mimic_patch import patch_g1_locomanip_mimic

            patch_g1_locomanip_mimic()

            # Set navigation p-controller for locomanip use case
            action_cfg = embodiment.get_action_cfg()
            # NOTE(mingxueg): set to false to avoid auto-nav without model output
            # if True, needs to set action_cfg.g1_action.navigation_subgoals
            action_cfg.g1_action.use_p_control = False

        scene = Scene(assets=[background, pick_up_object, destination_cart])
        isaaclab_arena_environment = IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=scene,
            task=G1TrayPickPlaceTask(pick_up_object, destination_cart, background, episode_length_s=30.0),
            teleop_device=teleop_device,
        )
        return isaaclab_arena_environment

    @staticmethod
    def add_cli_args(parser: argparse.ArgumentParser) -> None:
        parser.add_argument("--object", type=str, default="surgical_tray")
        parser.add_argument("--embodiment", type=str, default="g1_wbc_pink")
        parser.add_argument("--teleop_device", type=str, default=None)

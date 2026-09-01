# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Unitree G1 with Dex3 hands assembling a trocar in the LightWheel scene.

Builds through the AssetRegistry like the loco-manip scenes, but with a
different embodiment (dex hands, 37 DOF) and its own props, so it does not
share their base.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from i4h_arena.adapters.actuation import RobotSlice
from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config

#: Registered name of the dex-hand embodiment.
G1_DEX_EMBODIMENT = "g1_assemble_trocar_joint"

TROCAR_1_POSE = ((-1.60202, 1.91362, 0.87183), (-0.0, 0.70711, 0.70711, 0.0))
TROCAR_2_POSE = ((-1.50635, 1.90997, 0.8631), (-0.71475, -0.000243, 0.05853, 0.69692))
TRAY_POSE = ((-1.54919, 2.03365, 0.84554), (0.0, 0.0, -0.70711, 0.70711))


class G1TrocarScene(Scene):
    name = "g1_trocar"

    def register_assets(self) -> None:
        import i4h_arena.assets.g1_trocar  # noqa: F401,PLC0415
        import i4h_arena.embodiments.g1  # noqa: F401,PLC0415

    def build(self) -> Any:
        from isaaclab_arena.assets.asset_registry import AssetRegistry  # noqa: PLC0415
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415
        from isaaclab_arena.utils.pose import Pose as ArenaPose  # noqa: PLC0415

        from i4h_arena.envcfg.g1_trocar import G1TrocarEnvCfg  # noqa: PLC0415

        registry = AssetRegistry()
        background = registry.get_asset_by_name("trocar_assembly_scene")()
        trocar_1 = registry.get_asset_by_name("trocar_1")()
        trocar_2 = registry.get_asset_by_name("trocar_2")()
        tray = registry.get_asset_by_name("tray")()
        embodiment = registry.get_asset_by_name(G1_DEX_EMBODIMENT)(
            enable_cameras=bool(getattr(self.args, "enable_cameras", True))
        )

        trocar_1.set_initial_pose(ArenaPose(position_xyz=TROCAR_1_POSE[0], rotation_xyzw=TROCAR_1_POSE[1]))
        trocar_2.set_initial_pose(ArenaPose(position_xyz=TROCAR_2_POSE[0], rotation_xyzw=TROCAR_2_POSE[1]))
        tray.set_initial_pose(ArenaPose(position_xyz=TRAY_POSE[0], rotation_xyzw=TRAY_POSE[1]))

        steps = self.args.episode_steps or self.spec.max_steps
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=ArenaScene(assets=[background, trocar_1, trocar_2, tray]),
            # G1TrocarEnvCfg takes no env_spacing: the LightWheel scene is a
            # single authored room, not a tiled grid.
            task=G1TrocarEnvCfg(episode_length_s=max(15.0, (steps + 1) / self.spec.control_hz)),
        )

    def home_joints(self, env: Any) -> np.ndarray | None:
        home = get_robot_config(self.spec.embodiment).home_joint_pos_rad
        if not home:
            return None
        return np.tile(np.asarray(home, dtype=np.float32), (int(env.unwrapped.num_envs), 1))

    def robot_slices(self, env: Any) -> tuple[RobotSlice, ...]:
        width = int(env.action_space.shape[-1])
        return (RobotSlice("robot", 0, width, gripper_index=None),)

    def joint_orders(self) -> dict[str, tuple[str, ...]]:
        # The USD stores joints in articulation topology order, while both the
        # checkpoint state and the JointPositionAction term use this explicit
        # 29-body + 14-Dex3 order.
        from i4h_arena.envcfg.g1_trocar import ASSEMBLE_TROCAR_JOINT_NAMES  # noqa: PLC0415

        return {"robot": tuple(ASSEMBLE_TROCAR_JOINT_NAMES)}

    def camera_aliases(self) -> dict[str, str]:
        return {
            "front": "front_camera",
            "left_wrist": "left_wrist_camera",
            "right_wrist": "right_wrist_camera",
        }

    def object_aliases(self) -> dict[str, str]:
        return {
            "scene": "trocar_assembly_scene",
            "trocar": "trocar_1",
            "puncture_device": "trocar_2",
        }

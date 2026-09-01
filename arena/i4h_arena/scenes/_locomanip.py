# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared base for the Unitree G1 loco-manipulation scenes.

``g1_tray`` and ``g1_cart`` are the same Rheo room with a different prop.

These build through IsaacLab-Arena's ``AssetRegistry`` rather than an explicit
``make_assets()`` factory: their props are ``@register_asset``-decorated classes
that also need per-scene initial poses. The scissor and surgical scenes use the
factory form. Which applies is a property of how the assets were authored
upstream, not a choice this layer gets to make.
"""

from __future__ import annotations

from typing import Any

import numpy as np

from i4h_arena.adapters.actuation import RobotSlice
from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config

#: Registered whole-body controllers for policy/replay and teleop.
G1_POLICY_EMBODIMENT = "g1_wbc_joint"
G1_TELEOP_EMBODIMENT = "g1_wbc_pink"

Placement = tuple[tuple[float, float, float], tuple[float, float, float, float]]

#: Shared between both loco-manip scenes: (position_xyz, rotation_xyzw).
BACKGROUND_POSE: Placement = ((4.0, 0.0, -0.8), (0.0, 0.0, 0.0, 1.0))
CART_POSE: Placement = ((0.35, -1.65, -0.7875), (0.0, 0.0, 0.0, 1.0))


class LocomanipScene(Scene):
    """Unitree G1 standing in the Rheo room with one manipulable prop."""

    embodiment_name: str = G1_POLICY_EMBODIMENT
    pick_up_pose: Placement = ((-1.15, -1.6, -0.08), (0.0, 0.0, 0.707, 0.707))
    robot_pose: Placement = ((-0.5, -1.62, 0.0), (0.0, 0.0, 1.0, 0.0))

    def register_assets(self) -> None:
        # Side-effect imports: @register_asset populates the AssetRegistry.
        import i4h_arena.assets._locomanip  # noqa: F401,PLC0415
        import i4h_arena.embodiments.g1  # noqa: F401,PLC0415

    def build(self) -> Any:
        from isaaclab_arena.assets.asset_registry import AssetRegistry  # noqa: PLC0415
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415
        from isaaclab_arena.utils.pose import Pose as ArenaPose  # noqa: PLC0415

        registry = AssetRegistry()
        background = registry.get_asset_by_name("pre_op")()
        pick_up_object = registry.get_asset_by_name("surgical_tray")()
        destination_cart = registry.get_asset_by_name("cart")()
        embodiment_name = G1_TELEOP_EMBODIMENT if getattr(self.args, "mode", None) == "teleop" else self.embodiment_name
        embodiment = registry.get_asset_by_name(embodiment_name)(
            enable_cameras=bool(getattr(self.args, "enable_cameras", True))
        )

        background.set_initial_pose(ArenaPose(position_xyz=BACKGROUND_POSE[0], rotation_xyzw=BACKGROUND_POSE[1]))
        pick_up_object.set_initial_pose(
            ArenaPose(position_xyz=self.pick_up_pose[0], rotation_xyzw=self.pick_up_pose[1])
        )
        destination_cart.set_initial_pose(ArenaPose(position_xyz=CART_POSE[0], rotation_xyzw=CART_POSE[1]))
        embodiment.set_initial_pose(ArenaPose(position_xyz=self.robot_pose[0], rotation_xyzw=self.robot_pose[1]))

        assets = [background, pick_up_object, destination_cart]

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=ArenaScene(assets=assets),
            task=self.envcfg(pick_up_object, destination_cart, background),
        )

    def envcfg(self, pick_up_object: Any, destination_cart: Any, background: Any) -> Any:
        """The IsaacLab env cfg for this scene. Subclasses name their own."""
        raise NotImplementedError(f"{type(self).__name__} must supply an env cfg")

    def episode_length_s(self, minimum: float = 10.0) -> float:
        steps = self.args.episode_steps or self.spec.max_steps
        return max(minimum, (steps + 1) / self.spec.control_hz)

    def home_joints(self, env: Any) -> np.ndarray | None:
        home = get_robot_config(self.spec.embodiment).home_joint_pos_rad
        if not home:
            return None
        return np.tile(np.asarray(home, dtype=np.float32), (int(env.unwrapped.num_envs), 1))

    def robot_slices(self, env: Any) -> tuple[RobotSlice, ...]:
        # WBC consumes 43 measured joint targets followed by seven locomotion
        # command columns.  Policy inference writes all 50; scripted keyframes
        # write only the joint prefix and leave the command tail unchanged.
        width = int(env.action_space.shape[-1])
        joint_width = 43 if width >= 43 else None
        return (RobotSlice("robot", 0, width, gripper_index=None, joint_width=joint_width),)

    def camera_aliases(self) -> dict[str, str]:
        return {"head": "robot_head_cam"}

    def object_aliases(self) -> dict[str, str]:
        return {"background": "pre_op", "tray": "surgical_tray"}

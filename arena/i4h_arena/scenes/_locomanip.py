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

from i4h_arena.scenes.base import Scene

#: Registered whole-body controllers for policy/replay and teleop.
G1_POLICY_EMBODIMENT = "g1_locomanip_wbc_joint"
G1_TELEOP_EMBODIMENT = "g1_locomanip_wbc_pink"

Placement = tuple[tuple[float, float, float], tuple[float, float, float, float]]

#: Shared between both loco-manip scenes: (position_xyz, rotation_xyzw).
BACKGROUND_POSE: Placement = ((4.0, 0.0, -0.8), (0.0, 0.0, 0.0, 1.0))
CART_POSE: Placement = ((0.35, -1.65, -0.7875), (0.0, 0.0, 0.0, 1.0))


class LocomanipScene(Scene):
    """Unitree G1 standing in the Rheo room with one manipulable prop."""

    embodiment_name: str = G1_POLICY_EMBODIMENT
    pick_up_pose: Placement = ((-1.15, -1.6, -0.08), (0.0, 0.0, 0.707, 0.707))
    robot_pose: Placement = ((-0.5, -1.62, 0.0), (0.0, 0.0, 1.0, 0.0))

    @classmethod
    def add_cli_args(cls, parser: Any) -> None:
        parser.add_argument(
            "--embodiment",
            choices=(G1_POLICY_EMBODIMENT, G1_TELEOP_EMBODIMENT),
            default=G1_POLICY_EMBODIMENT,
            help="G1 whole-body controller: joint targets for policies and replay (default), "
            "or Pink IK wrist poses for keyboard_23d teleoperation.",
        )

    def register_assets(self) -> None:
        # Side-effect imports: @register_asset populates the AssetRegistry.
        import i4h_arena.assets._locomanip  # noqa: F401,PLC0415
        import i4h_arena.embodiments.g1  # noqa: F401,PLC0415

    def build(self) -> Any:
        from isaaclab_arena.assets.registries import AssetRegistry  # noqa: PLC0415
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.environments.isaaclab_arena_manager_based_env_cfg import (  # noqa: PLC0415
            set_control_rate_50hz,
        )
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415
        from isaaclab_arena.utils.pose import Pose as ArenaPose  # noqa: PLC0415

        registry = AssetRegistry()
        background = registry.get_asset_by_name("pre_op")()
        pick_up_object = registry.get_asset_by_name("surgical_tray")()
        destination_cart = registry.get_asset_by_name("cart")()
        embodiment_name = getattr(self.args, "embodiment", None) or self.embodiment_name
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
            # The G1 WBC and GR00T policies run at 50 Hz, Arena's rate before 0.3.
            env_cfg_callback=set_control_rate_50hz,
        )

    def envcfg(self, pick_up_object: Any, destination_cart: Any, background: Any) -> Any:
        """The IsaacLab env cfg for this scene. Subclasses name their own."""
        raise NotImplementedError(f"{type(self).__name__} must supply an env cfg")

    def configure_env_cfg(self, env_cfg: Any) -> None:
        if getattr(self.args, "embodiment", None) != G1_TELEOP_EMBODIMENT:
            return
        from isaaclab.devices.device_base import DevicesCfg  # noqa: PLC0415

        from i4h_arena.devices.g1_keyboard import G1Keyboard23DCfg  # noqa: PLC0415

        # Isaac Lab's record_demos.py --teleop_device keyboard_23d finds the device here.
        env_cfg.teleop_devices = DevicesCfg(devices={"keyboard_23d": G1Keyboard23DCfg(sim_device=env_cfg.sim.device)})

    def episode_length_s(self, minimum: float = 10.0) -> float:
        steps = self.args.episode_steps or self.spec.max_steps
        return max(minimum, (steps + 1) / self.spec.control_hz)

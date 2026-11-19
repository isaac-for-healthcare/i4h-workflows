# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Surface cloth and bimanual Pink controllers recovered from Rheo v0.7.0."""

from __future__ import annotations

from datetime import datetime
from typing import Any

from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config
from i4h_common.paths import workflow_root


class TableclothScene(Scene):
    robot_kind = "g1"

    def configure_args(self, args: Any) -> None:
        if args.num_envs != 1:
            raise ValueError("tablecloth supports one environment per XR/recording session")
        args.presets = args.presets or "newton"
        if args.presets != "newton":
            raise ValueError("tablecloth requires --presets newton; PhysX cannot create a valid cloth view")
        args.enable_pinocchio = True
        super().configure_args(args)

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment  # noqa: PLC0415
        from isaaclab_arena.scene.scene import Scene as ArenaScene  # noqa: PLC0415
        from isaaclab_arena.utils.physics_backend import PhysicsBackend  # noqa: PLC0415

        from i4h_arena.envcfg.tablecloth.cloth_physics import select_physics_backend  # noqa: PLC0415
        from i4h_arena.envcfg.tablecloth.environment import (  # noqa: PLC0415
            TableclothEmbodiment,
            TableclothTask,
            room_assets,
        )

        if self.robot_kind == "g1":
            from i4h_arena.envcfg.tablecloth.g1_spread_tablecloth_teleop_env_cfg import G1SpreadTableclothTeleopEnvCfg

            cfg = G1SpreadTableclothTeleopEnvCfg()
        else:
            from i4h_arena.envcfg.tablecloth.h2_spread_tablecloth_teleop_env_cfg import H2SpreadTableclothTeleopEnvCfg

            cfg = H2SpreadTableclothTeleopEnvCfg()
        device = getattr(self.args, "device", None) or "cuda:0"
        cfg.sim.device = device
        if self.robot_kind == "g1":
            # Pink IK writes the robot's URDF where it runs: one directory per run of this scene.
            run_dir = workflow_root() / "runs" / self.name / datetime.now().strftime("%Y%m%d_%H%M%S")
            cfg.actions.pink_ik_cfg.controller.urdf_output_dir = str(run_dir / "robot")
        cfg.isaac_teleop.sim_device = device
        # Cloth needs Newton: the environment defaults to it, and --presets physx is rejected here.
        select_physics_backend(cfg, getattr(self.args, "presets", None) or "newton")
        if self.robot_kind == "g1":
            from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

            # The recovered fixed-base G1 disables robot gravity via PhysX.
            # MJWarp consumes mjc:gravcomp instead; cloth gravity stays on. Isaac Lab
            # 3.0 GA registers MuJoCo's USD attributes for the coupled solver itself.
            cfg.scene.robot.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
        no_cameras = getattr(self.args, "no_cameras", not getattr(self.args, "enable_cameras", True))
        cameras = () if no_cameras else tuple(self.camera_aliases().values())
        if no_cameras:
            cfg.observations.camera_images = None
        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=TableclothEmbodiment(cfg, cameras, get_robot_config(self.spec.embodiment).joint_names),
            scene=ArenaScene(assets=room_assets(cfg, ("robot", *self.camera_aliases().values()))),
            task=TableclothTask(cfg),
            default_physics_backend=PhysicsBackend.NEWTON,
        )

    def camera_aliases(self) -> dict[str, str]:
        aliases = {"front": "front_camera"}
        if self.robot_kind == "g1":
            aliases.update(left_wrist="left_wrist_camera", right_wrist="right_wrist_camera")
        return aliases

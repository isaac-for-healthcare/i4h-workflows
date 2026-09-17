# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Surface cloth and bimanual Pink controllers recovered from Rheo v0.7.0."""

from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from i4h_arena.adapters.scene_view import ArenaSceneView, _np
from i4h_arena.scenes.base import Scene
from i4h_common.config import get_robot_config
from i4h_common.paths import workflow_root
from i4h_common.types import ObjectState, Pose


class TableclothView(ArenaSceneView):
    def bimanual_state(self) -> np.ndarray:
        """Measured wrists (XYZ/XYZW) and fingers in the controller's order."""
        data = self._scene["robot"].data
        poses = _np(data.body_link_pose_w)
        origins = _np(self._scene.env_origins)
        wrists = []
        for name in ("left_wrist_yaw_link", "right_wrist_yaw_link"):
            pose = poses[:, data.body_names.index(name)].copy()
            pose[:, :3] -= origins
            wrists.append(pose)
        names = self._env.unwrapped.cfg.actions.pink_ik_cfg.hand_joint_names
        ids = [data.joint_names.index(name) for name in names]
        return np.concatenate([*wrists, _np(data.joint_pos)[:, ids]], axis=-1)

    def object(self, name: str) -> ObjectState:
        if name != "cloth":
            return super().object(name)
        # A surface mesh has no rigid orientation. Expose its simulated
        # centroid/velocity; identity denotes the world-aligned centroid frame.
        data = self._scene["cloth"].data
        return ObjectState(
            name="cloth",
            pose=Pose(pos=_np(data.root_pos_w), quat=Pose.identity(self.num_envs).quat),
            lin_vel=_np(data.root_vel_w),
            ang_vel=np.zeros((self.num_envs, 3), dtype=np.float32),
        )

    def teleop_config(self) -> Any:
        """Scene-owned retargeting configuration, consumed by the XR input adapter."""
        return self._env.unwrapped.cfg.isaac_teleop


class TableclothScene(Scene):
    robot_kind = "g1"

    def configure_args(self, args: Any) -> None:
        if args.num_envs != 1:
            raise ValueError("tablecloth supports one environment per XR/recording session")
        args.presets = args.presets or "newton"
        if args.presets != "newton":
            raise ValueError("tablecloth requires --presets newton; PhysX cannot create a valid cloth view")
        # The pinned USD physics parser races on bodies with many colliders.
        # AppContext preserves this limit across Kit's bootstrap override.
        args.usd_thread_limit = 1
        args.enable_pinocchio = True
        args.xr = args.mode == "teleop"
        if args.xr and args.headless:
            raise ValueError("XR tablecloth teleoperation requires a visible Kit session")
        super().configure_args(args)

    def build(self) -> Any:
        # These are direct IsaacLab configurations. The Scene still supplies
        # the normal adapters and the shared SimulationRunner owns every step.
        from i4h_arena.envcfg.tablecloth.cloth_physics import select_physics_backend

        if self.robot_kind == "g1":
            from i4h_arena.envcfg.tablecloth.g1_spread_tablecloth_teleop_env_cfg import G1SpreadTableclothTeleopEnvCfg

            cfg = G1SpreadTableclothTeleopEnvCfg()
        else:
            from i4h_arena.envcfg.tablecloth.h2_spread_tablecloth_teleop_env_cfg import H2SpreadTableclothTeleopEnvCfg

            cfg = H2SpreadTableclothTeleopEnvCfg()
        cfg.sim.device = self.args.device
        if self.robot_kind == "g1":
            run_dir = Path(
                os.environ.get("I4H_RUN_DIR")
                or (workflow_root() / "runs" / self.args.workflow / datetime.now().strftime("%Y%m%d_%H%M%S"))
            )
            cfg.actions.pink_ik_cfg.controller.urdf_output_dir = str(run_dir / "robot")
        cfg.isaac_teleop.sim_device = self.args.device
        cfg.scene.num_envs = 1
        cfg.recorders = None
        cfg.terminations.time_out = None
        select_physics_backend(cfg, self.args.presets)
        if self.robot_kind == "g1":
            from isaaclab_newton.sim.schemas import MujocoRigidBodyPropertiesCfg

            from i4h_arena.envcfg.tablecloth.cloth_physics import enable_mujoco_usd_attributes

            # The recovered fixed-base G1 disables robot gravity via PhysX.
            # MJWarp consumes mjc:gravcomp instead; cloth gravity stays on.
            cfg.scene.robot.spawn.rigid_props = MujocoRigidBodyPropertiesCfg(gravcomp=1.0)
            enable_mujoco_usd_attributes()
        if self.args.no_cameras:
            for name in self.camera_aliases().values():
                setattr(cfg.scene, name, None)
            cfg.observations.camera_images = None
        self._cfg = cfg
        return cfg

    def gym_spec(self) -> tuple[str, Any]:
        import gymnasium as gym

        gym_id = f"I4H-{self.name}-v0"
        if gym_id not in gym.registry:
            gym.register(gym_id, entry_point="isaaclab.envs:ManagerBasedRLEnv", disable_env_checker=True)
        return gym_id, self.build()

    def joint_orders(self) -> dict[str, tuple[str, ...]]:
        return {"robot": get_robot_config(self.spec.embodiment).joint_names}

    def camera_aliases(self) -> dict[str, str]:
        aliases = {"front": "front_camera"}
        if self.robot_kind == "g1":
            aliases.update(left_wrist="left_wrist_camera", right_wrist="right_wrist_camera")
        return aliases

    def make_view(self, env: Any) -> TableclothView:
        return TableclothView(
            env,
            objects=self.spec.objects,
            cameras=self.spec.cameras,
            gripper=False,
            joint_orders=self.joint_orders(),
            camera_aliases=self.camera_aliases(),
        )

    def make_actuation(self, env: Any, view: ArenaSceneView | None = None) -> Any:
        act = super().make_actuation(env, view)
        act.seed((view or self.make_view(env)).bimanual_state())
        return act

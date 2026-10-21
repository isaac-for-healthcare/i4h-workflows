# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.tasks.predicates.spatial import objects_in_proximity
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from i4h_arena.envcfg._locomanip import LocomanipTask, randomize_pose_event, subtask
from i4h_arena.tensor_utils import to_torch


class G1TrayEnvCfg(LocomanipTask):
    """Rheo tray pick-and-place task in the pre-operative scene."""

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([1.0, -1.0, 1.5]),
        )

    def get_mimic_env_cfg(self, embodiment_name: str):
        tray = self.pick_up_object.name
        return self.mimic_env_cfg(
            "g1_tray_pick_and_place_task_D0",
            {
                "right": [
                    subtask(tray, "right_before_grasp_box", 0.001),
                    subtask(tray, "right_after_release_box", 0.001),
                    subtask(tray),
                ],
                "left": [
                    subtask(tray, "left_before_grasp_box", 0.001),
                    subtask(tray, "left_after_release_box", 0.001),
                    subtask(tray),
                ],
                "body": [
                    subtask(tray, "body_face_box_front", 0.001),
                    subtask(tray, "body_face_cart_front", 0.001),
                    subtask(tray),
                ],
            },
        )

    def get_events_cfg(self) -> TrayEventsCfg:
        return TrayEventsCfg(
            reset_success_state=EventTermCfg(func=reset_tray_success_state, mode="reset"),
            reset_destination_cart_pose=randomize_pose_event(self.destination_cart, x=(-0.01, 0.01), y=(-0.01, 0.01)),
            reset_pick_up_object_pose=randomize_pose_event(self.pick_up_object, x=(-0.025, 0.025), y=(-0.025, 0.025)),
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        placed = TerminationTermCfg(
            func=tray_placed_success,
            params={
                "object_cfg": SceneEntityCfg(self.pick_up_object.name),
                "target_object_cfg": SceneEntityCfg(self.destination_cart.name),
                "max_x_separation": 0.16,
                "max_y_separation": 0.16,
                "max_z_separation": 0.88,
                "settle_steps": 20,
                "max_linear_speed": 0.08,
            },
        )
        return self.termination_cfg("tray_placed", placed)


def tray_placed_success(
    env,
    object_cfg: SceneEntityCfg,
    target_object_cfg: SceneEntityCfg,
    max_x_separation: float = 0.16,
    max_y_separation: float = 0.16,
    max_z_separation: float = 0.88,
    settle_steps: int = 20,
    max_linear_speed: float = 0.08,
) -> torch.Tensor:
    """Success only after the tray remains near the cart and stops moving."""
    placed = objects_in_proximity(
        env,
        object_cfg=object_cfg,
        target_object_cfg=target_object_cfg,
        max_x_separation=max_x_separation,
        max_y_separation=max_y_separation,
        max_z_separation=max_z_separation,
    )
    speed = _root_linear_speed(env, object_cfg)
    stable = speed <= max_linear_speed
    ready = placed & stable
    if not hasattr(env, "_tray_success_settle_count"):
        env._tray_success_settle_count = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
    env._tray_success_settle_count = torch.where(
        ready,
        env._tray_success_settle_count + 1,
        torch.zeros_like(env._tray_success_settle_count),
    )
    return env._tray_success_settle_count >= settle_steps


def _root_linear_speed(env, asset_cfg: SceneEntityCfg) -> torch.Tensor:
    data = env.scene[asset_cfg.name].data
    return torch.linalg.norm(to_torch(data.root_lin_vel_w), dim=-1)


def reset_tray_success_state(env, env_ids: torch.Tensor) -> None:
    if hasattr(env, "_tray_success_settle_count"):
        env._tray_success_settle_count[env_ids] = 0


@configclass
class TrayEventsCfg:
    reset_success_state: EventTermCfg = MISSING
    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING

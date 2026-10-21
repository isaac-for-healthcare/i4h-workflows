# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object

from i4h_arena.envcfg._locomanip import LocomanipTask, randomize_pose_event, subtask
from i4h_arena.tensor_utils import to_torch


def cart_displaced(env: ManagerBasedRLEnv, cart_cfg: SceneEntityCfg, minimum_distance: float) -> torch.Tensor:
    """Succeed after the cart has moved ``minimum_distance`` from where the reset placed it.

    The origin is recorded by :func:`record_cart_reset_position`, a reset event, because
    Arena's progress tracker evaluates this predicate without resetting it between episodes.
    """
    current_xy = to_torch(env.scene[cart_cfg.name].data.root_pos_w)[:, :2]
    origin = getattr(env, "_i4h_cart_reset_xy", None)
    if origin is None:
        origin = env._i4h_cart_reset_xy = current_xy.clone()
    return torch.linalg.vector_norm(current_xy - origin, dim=-1) >= minimum_distance


def record_cart_reset_position(env: ManagerBasedRLEnv, env_ids: torch.Tensor | None, cart_cfg: SceneEntityCfg) -> None:
    """Remember where this episode's reset left the cart; ordered after the cart pose reset."""
    current_xy = to_torch(env.scene[cart_cfg.name].data.root_pos_w)[:, :2]
    origin = getattr(env, "_i4h_cart_reset_xy", None)
    if origin is None:
        env._i4h_cart_reset_xy = current_xy.clone()
        return
    ids = slice(None) if env_ids is None else env_ids
    origin[ids] = current_xy[ids]


class G1CartEnvCfg(LocomanipTask):
    """Rheo cart-push task in the pre-operative scene."""

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([1.0, 0.5, 1.0]),
        )

    def get_mimic_env_cfg(self, embodiment_name: str):
        cart = self.destination_cart.name
        return self.mimic_env_cfg(
            "g1_push_cart_task_D0",
            {
                "right": [subtask(cart, "right_before_grasp_cart", 0.001), subtask(cart)],
                "left": [subtask(cart, "left_before_grasp_cart", 0.001), subtask(cart)],
                "body": [
                    subtask(cart, "body_face_cart", 0.001),
                    subtask(cart, "body_pushing_cart", 0.001),
                    subtask(cart),
                ],
            },
        )

    def get_events_cfg(self) -> CartEventsCfg:
        return CartEventsCfg(
            reset_destination_cart_pose=randomize_pose_event(self.destination_cart, x=(-0.01, 0.03), y=(0.01, 0.04)),
            reset_pick_up_object_pose=randomize_pose_event(
                self.pick_up_object, x=(-0.025, 0.025), y=(-0.025, 0.025), yaw=(-0.3, 0.3)
            ),
            record_cart_reset_position=EventTermCfg(
                func=record_cart_reset_position,
                mode="reset",
                params={"cart_cfg": SceneEntityCfg(self.destination_cart.name)},
            ),
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        pushed = TerminationTermCfg(
            func=cart_displaced,
            params={"cart_cfg": SceneEntityCfg(self.destination_cart.name), "minimum_distance": 2.0},
        )
        return self.termination_cfg("push_cart", pushed)


@configclass
class CartEventsCfg:
    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING
    record_cart_reset_position: EventTermCfg = MISSING

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

from dataclasses import MISSING

import isaaclab.envs.mdp as mdp_isaac_lab
import numpy as np
import torch
from isaaclab.envs import ManagerBasedRLEnv
from isaaclab.envs.common import ViewerCfg
from isaaclab.envs.mimic_env_cfg import SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils import configclass
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.cameras import get_viewer_cfg_look_at_object
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events

from i4h_arena.envcfg._locomanip import G1LocomanipPickAndPlaceTask
from i4h_arena.rendering import QUALITY, merge_rtx_settings
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


class G1CartEnvCfg(G1LocomanipPickAndPlaceTask):
    """Rheo cart-push task in the pre-operative scene."""

    #: Isaac Lab's former "quality" rendering mode with DLAA. Merged into the scene's RTX
    #: settings by :meth:`i4h_arena.scenes.base.Scene.arena_environment`.
    rtx_settings = merge_rtx_settings(QUALITY, {"antialiasing_mode": "DLAA"})

    def get_viewer_cfg(self) -> ViewerCfg:
        return get_viewer_cfg_look_at_object(
            lookat_object=self.pick_up_object,
            offset=np.array([1.0, 0.5, 1.0]),
        )

    def get_mimic_env_cfg(self, embodiment_name: str):
        env_cfg = super().get_mimic_env_cfg(embodiment_name)
        env_cfg.datagen_config.name = "g1_push_cart_task_D0"
        env_cfg.subtask_configs = {
            "right": [
                _subtask("cart", "right_before_grasp_cart", 0.001),
                _subtask("cart", None, 0.0),
            ],
            "left": [
                _subtask("cart", "left_before_grasp_cart", 0.001),
                _subtask("cart", None, 0.0),
            ],
            "body": [
                _subtask("cart", "body_face_cart", 0.001),
                _subtask("cart", "body_pushing_cart", 0.001),
                _subtask("cart", None, 0.0),
            ],
        }
        return env_cfg

    def get_events_cfg(self):
        return I4HEventsCfg(pick_up_object=self.pick_up_object, destination_cart=self.destination_bin)

    def get_termination_cfg(self) -> TaskTerminationCfg:
        pushed = TerminationTermCfg(
            func=cart_displaced,
            params={
                "cart_cfg": SceneEntityCfg(self.destination_bin.name),
                "minimum_distance": 2.0,
            },
        )
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TaskTerminationCfg(
            timeout_s=self.get_episode_length_s(),
            success=[CompletionCriteria(name="push_cart", predicate_sequence=[pushed])],
            failures={"object_dropped": object_dropped},
        )



def _subtask(object_ref: str, signal: str | None, action_noise: float) -> SubTaskConfig:
    kwargs = {"subtask_term_signal": signal} if signal is not None else {}
    return SubTaskConfig(
        object_ref=object_ref,
        first_subtask_start_offset_range=(0, 0),
        subtask_term_offset_range=(0, 0),
        selection_strategy="nearest_neighbor_object",
        selection_strategy_kwargs={"nn_k": 3},
        action_noise=action_noise,
        num_interpolation_steps=0,
        num_fixed_steps=0,
        apply_noise_during_interpolation=False,
        **kwargs,
    )


@configclass
class I4HEventsCfg:
    reset_destination_cart_pose: EventTermCfg = MISSING
    reset_pick_up_object_pose: EventTermCfg = MISSING
    record_cart_reset_position: EventTermCfg = MISSING

    def __init__(self, pick_up_object: Asset, destination_cart: Asset):
        object_initial_pose = pick_up_object.get_initial_pose()
        object_roll, object_pitch, object_yaw = euler_xyz_from_quat(
            torch.tensor(object_initial_pose.rotation_xyzw).reshape(1, 4)
        )
        self.reset_pick_up_object_pose = EventTermCfg(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (object_initial_pose.position_xyz[0] - 0.025, object_initial_pose.position_xyz[0] + 0.025),
                    "y": (object_initial_pose.position_xyz[1] - 0.025, object_initial_pose.position_xyz[1] + 0.025),
                    "z": (object_initial_pose.position_xyz[2], object_initial_pose.position_xyz[2]),
                    "roll": (object_roll, object_roll),
                    "pitch": (object_pitch, object_pitch),
                    "yaw": (object_yaw - 0.3, object_yaw + 0.3),
                },
                "asset_cfgs": [SceneEntityCfg(pick_up_object.name)],
            },
        )

        cart_initial_pose = destination_cart.get_initial_pose()
        cart_roll, cart_pitch, cart_yaw = euler_xyz_from_quat(
            torch.tensor(cart_initial_pose.rotation_xyzw).reshape(1, 4)
        )
        self.reset_destination_cart_pose = EventTermCfg(
            func=franka_stack_events.randomize_object_pose,
            mode="reset",
            params={
                "pose_range": {
                    "x": (cart_initial_pose.position_xyz[0] - 0.01, cart_initial_pose.position_xyz[0] + 0.03),
                    "y": (cart_initial_pose.position_xyz[1] + 0.01, cart_initial_pose.position_xyz[1] + 0.04),
                    "z": (cart_initial_pose.position_xyz[2], cart_initial_pose.position_xyz[2]),
                    "roll": (cart_roll, cart_roll),
                    "pitch": (cart_pitch, cart_pitch),
                    "yaw": (cart_yaw, cart_yaw),
                },
                "asset_cfgs": [SceneEntityCfg(destination_cart.name)],
            },
        )
        self.record_cart_reset_position = EventTermCfg(
            func=record_cart_reset_position,
            mode="reset",
            params={"cart_cfg": SceneEntityCfg(destination_cart.name)},
        )

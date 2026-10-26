# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the G1 cart-push and tray pick-and-place tasks share.

Both are Arena tasks in the Rheo pre-operative room with a surgical tray and a cart.
Arena's ``PickAndPlaceTask`` does not fit them: it judges success by settle, lift and
contact-supported placement, while these tasks keep the success checks their GR00T
policies were validated against, and pushing a cart is not a pick-and-place.
"""

from __future__ import annotations

import isaaclab.envs.mdp as mdp_isaac_lab
import torch
from isaaclab.envs.mimic_env_cfg import MimicEnvCfg, SubTaskConfig
from isaaclab.managers import EventTermCfg, SceneEntityCfg, TerminationTermCfg
from isaaclab.utils.math import euler_xyz_from_quat
from isaaclab_arena.assets.asset import Asset
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.metrics.success_rate import SuccessRateMetric
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_tasks.contrib.stack.mdp import franka_stack_events

from i4h_arena.rendering import QUALITY, merge_rtx_settings


class LocomanipTask(TaskBase):
    """A G1 goal between a pick-up object and a destination cart."""

    #: Isaac Lab's former "quality" rendering mode with DLAA. Merged into the scene's RTX
    #: settings by :meth:`i4h_arena.scenes.base.Scene.arena_environment`.
    rtx_settings = merge_rtx_settings(QUALITY, {"antialiasing_mode": "DLAA"})

    #: The instruction the task's GR00T policy was trained on. Arena's policy runner hands the
    #: task description to the policy, so it must be the trained prompt word for word.
    instruction: str = ""

    def __init__(
        self, pick_up_object: Asset, destination_cart: Asset, background_scene: Asset, episode_length_s: float
    ) -> None:
        super().__init__(episode_length_s=episode_length_s)
        self.pick_up_object = pick_up_object
        self.destination_cart = destination_cart
        self.background_scene = background_scene
        self.task_description = (
            self.instruction or f"Pick up the {pick_up_object.name}, and place it into the {destination_cart.name}"
        )

    def get_scene_cfg(self) -> None:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return [SuccessRateMetric()]

    def termination_cfg(self, name: str, success: TerminationTermCfg) -> TaskTerminationCfg:
        """Succeed on ``success``; fail when the pick-up object falls below the room."""
        object_dropped = TerminationTermCfg(
            func=mdp_isaac_lab.root_height_below_minimum,
            params={
                "minimum_height": self.background_scene.object_min_z,
                "asset_cfg": SceneEntityCfg(self.pick_up_object.name),
            },
        )
        return TaskTerminationCfg(
            timeout_s=self.get_episode_length_s(),
            success=[CompletionCriteria(name=name, predicate_sequence=[success])],
            failures={"object_dropped": object_dropped},
        )

    @staticmethod
    def mimic_env_cfg(name: str, subtask_configs: dict[str, list[SubTaskConfig]]) -> MimicEnvCfg:
        """Isaac Lab Mimic config with the G1 loco-manipulation data-generation settings."""
        cfg = MimicEnvCfg()
        datagen = cfg.datagen_config
        datagen.name = name
        datagen.generation_guarantee = True
        datagen.generation_keep_failed = False
        datagen.generation_num_trials = 100
        datagen.generation_select_src_per_subtask = False
        datagen.generation_select_src_per_arm = False
        datagen.generation_relative = False
        datagen.generation_joint_pos = False
        datagen.generation_transform_first_robot_pose = False
        datagen.generation_interpolate_from_last_target_pose = True
        datagen.max_num_failures = 25
        datagen.seed = 1
        cfg.subtask_configs = subtask_configs
        return cfg


def subtask(object_ref: str, signal: str | None = None, action_noise: float = 0.0) -> SubTaskConfig:
    """One Mimic subtask relative to ``object_ref``; the final subtask of an arm has no signal."""
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


def randomize_pose_event(
    asset: Asset,
    x: tuple[float, float] = (0.0, 0.0),
    y: tuple[float, float] = (0.0, 0.0),
    yaw: tuple[float, float] = (0.0, 0.0),
) -> EventTermCfg:
    """Reset event placing ``asset`` uniformly within offsets of its initial pose [m, rad]."""
    pose = asset.get_initial_pose()
    roll0, pitch0, yaw0 = euler_xyz_from_quat(torch.tensor(pose.rotation_xyzw).reshape(1, 4))
    x0, y0, z0 = pose.position_xyz
    return EventTermCfg(
        func=franka_stack_events.randomize_object_pose,
        mode="reset",
        params={
            "pose_range": {
                "x": (x0 + x[0], x0 + x[1]),
                "y": (y0 + y[0], y0 + y[1]),
                "z": (z0, z0),
                "roll": (roll0, roll0),
                "pitch": (pitch0, pitch0),
                "yaw": (yaw0 + yaw[0], yaw0 + yaw[1]),
            },
            "asset_cfgs": [SceneEntityCfg(asset.name)],
        },
    )

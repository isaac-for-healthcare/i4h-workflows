# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A spread-tablecloth env cfg as Arena's embodiment, scene assets and task.

The G1 and H2 env cfgs stay the single description of each world. These classes
hand its parts to Arena, which composes the env like every other i4h scene.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

from isaaclab.scene import InteractiveSceneCfg
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.metrics.metric_base import MetricBase
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.configclass import make_configclass

from i4h_arena.assets.config_asset import ConfigAsset


class TableclothEmbodiment(EmbodimentBase):
    """The env cfg's robot and cameras, its Pink IK action, observations and XR anchor."""

    name = "tablecloth_robot"
    tags = ["embodiment"]

    def __init__(self, env_cfg: Any, camera_names: Iterable[str]) -> None:
        super().__init__(enable_cameras=True)
        fields = [("robot", type(env_cfg.scene.robot), env_cfg.scene.robot)]
        for name in camera_names:
            camera = getattr(env_cfg.scene, name)
            fields.append((name, type(camera), camera))
        self.scene_config = make_configclass("TableclothRobotSceneCfg", fields)()
        self.camera_config = None
        self.action_config = env_cfg.actions
        self.observation_config = env_cfg.observations
        self.event_config = None
        self.xr = env_cfg.xr

    def get_recorder_term_cfg(self, record_trajectories: bool = False) -> Any:
        from i4h_arena.embodiments.recorders import action_state_recorders  # noqa: PLC0415

        return action_state_recorders(super().get_recorder_term_cfg(record_trajectories))


class TableclothTask(TaskBase):
    """Spread the cloth: the env cfg's reset event and time-out, with no success term.

    Arena composes managers, not the simulation or teleop settings, so the task's
    ``tune_env_cfg`` carries those over from the env cfg.
    """

    def __init__(self, env_cfg: Any) -> None:
        super().__init__(episode_length_s=env_cfg.episode_length_s)
        self._env_cfg = env_cfg

    def get_scene_cfg(self) -> None:
        return None

    def get_events_cfg(self) -> Any:
        return self._env_cfg.events

    def get_termination_cfg(self) -> TaskTerminationCfg:
        return TaskTerminationCfg(timeout_s=self.get_episode_length_s())

    def get_mimic_env_cfg(self, arm_mode: Any) -> None:
        return None

    def get_metrics(self) -> list[MetricBase]:
        return []

    def get_viewer_cfg(self) -> Any:
        return self._env_cfg.viewer

    def tune_env_cfg(self, env_cfg: Any) -> Any:
        source = self._env_cfg
        env_cfg.decimation = source.decimation
        env_cfg.sim.dt = source.sim.dt
        env_cfg.sim.render_interval = source.sim.render_interval
        env_cfg.sim.physics = source.sim.physics
        env_cfg.scene.env_spacing = source.scene.env_spacing
        env_cfg.scene.replicate_physics = source.scene.replicate_physics
        env_cfg.isaac_teleop = source.isaac_teleop
        env_cfg.wait_for_textures = source.wait_for_textures
        return env_cfg


def room_assets(env_cfg: Any, robot_fields: Iterable[str]) -> list[ConfigAsset]:
    """One Arena asset per scene entity that is not the robot or one of its cameras."""
    skip = set(robot_fields) | set(InteractiveSceneCfg.__dataclass_fields__)
    return [
        ConfigAsset(name, value)
        for name, value in vars(env_cfg.scene).items()
        if name not in skip and value is not None
    ]

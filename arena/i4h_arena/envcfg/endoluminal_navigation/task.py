# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The catheter navigation goal as an Arena task.

:mod:`i4h_arena.envcfg.endoluminal_navigation.terms` builds the IsaacLab term configs
and stays importable without Arena; this module hands those terms to Arena's
environment builder, which merges them with the scene and the embodiment.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Any

from isaaclab.envs.common import ViewerCfg
from isaaclab_arena.embodiments.common.arm_mode import ArmMode
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg

from i4h_arena.envcfg.endoluminal_navigation.terms import (
    CatheterNavigationEventsCfg,
    CatheterNavigationObservationsCfg,
    CatheterNavigationRewardsCfg,
    navigation_observations_cfg,
    navigation_rewards_cfg,
    navigation_terminations_cfg,
)


class CatheterNavigationTask(TaskBase):
    """Reach the far end of the patient's route: the endoluminal scenes' Arena task.

    The route comes from the patient twin and is resolved while the embodiment
    seeds the rod along it, so the task is built from that resolved route
    rather than deriving a second copy. Without a twin there is no vessel to
    navigate and every term getter returns ``None``: a workflow reading
    ``success`` then gets the absent-term zeros instead of a goal it can never
    meet, and the remaining arc and lumen width the rewards measure are
    undefined. Teleop and replay drive the action terms directly and never read
    the observation or reward managers, so they are unaffected either way.

    ``episode_length_s`` is the step budget the scene manifest validates, as time. The
    task owns it so that every consumer, including the RL trainers that register this
    environment without the workflow scene's ``configure_env_cfg``, truncates at it.

    ``flat_observations`` selects the online RSL-RL variant. The navigation
    group normally leaves its terms unconcatenated and carries the fluoroscopy
    view, because the RLinf bridge composes GR00T's modality dict out of named
    keys. An RSL-RL actor wants a single vector and cannot concatenate an image
    with fifteen scalars, so that variant drops the view and flattens the rest.
    Dropping the image is the point rather than a concession: the RSL-RL
    profiles exist to ask whether the navigation reward is learnable from the
    geometry alone.
    """

    def __init__(
        self,
        target_world_m: Iterable[float] | None,
        *,
        route_world_m: Sequence[Sequence[float]] | None,
        lumen_radii_m: Sequence[float] | None = None,
        episode_length_s: float | None = None,
        flat_observations: bool = False,
    ) -> None:
        super().__init__(episode_length_s=episode_length_s)
        self.target_world_m = None if target_world_m is None else tuple(float(value) for value in target_world_m)
        self.route_world_m = route_world_m
        self.lumen_radii_m = lumen_radii_m
        self.flat_observations = flat_observations

    @classmethod
    def for_embodiment(
        cls, embodiment: Any, *, episode_length_s: float | None = None, flat_observations: bool = False
    ) -> CatheterNavigationTask:
        """Bind the goal to the route and target a catheter embodiment seeded its rod on."""
        return cls(
            embodiment.navigation_target_world_m,
            route_world_m=embodiment.rod_spec.initial_path_world_m,
            lumen_radii_m=embodiment.rod_spec.lumen_radii_m,
            episode_length_s=episode_length_s,
            flat_observations=flat_observations,
        )

    def get_scene_cfg(self) -> None:
        return None

    def get_observation_cfg(self) -> CatheterNavigationObservationsCfg | None:
        """Low-dimensional navigation state, bound to this scene's route and target."""
        if self.target_world_m is None:
            return None
        config = navigation_observations_cfg(self.target_world_m, route_world_m=self.route_world_m)
        if self.flat_observations:
            config.policy.fluoroscopy_rgb = None
            config.policy.concatenate_terms = True
        return config

    def get_events_cfg(self) -> CatheterNavigationEventsCfg | None:
        if self.target_world_m is None:
            return None
        return CatheterNavigationEventsCfg()

    def get_rewards_cfg(self) -> CatheterNavigationRewardsCfg | None:
        """Dense navigation objective, bound to this scene's own route and wall."""
        if self.target_world_m is None:
            return None
        return navigation_rewards_cfg(
            self.target_world_m,
            route_world_m=self.route_world_m,
            lumen_radii_m=self.lumen_radii_m,
        )

    def get_termination_cfg(self) -> TaskTerminationCfg:
        """Report arrival at the distal centerline as success, and the step budget as truncation."""
        if self.target_world_m is None:
            return TaskTerminationCfg(timeout_s=None)
        # The same polyline the rod was seeded along, so the readout measures
        # remaining vessel against the route the target is the far end of.
        arrival = navigation_terminations_cfg(self.target_world_m, route_world_m=self.route_world_m).success
        return TaskTerminationCfg(
            timeout_s=self.get_episode_length_s(),
            success=[CompletionCriteria(name="arrival", predicate_sequence=[arrival])],
        )

    def get_mimic_env_cfg(self, arm_mode: ArmMode) -> None:
        return None

    def get_metrics(self) -> list:
        return []

    def get_viewer_cfg(self) -> ViewerCfg:
        # The view an Arena environment without a task opens on, which is what
        # these scenes used before they had one. The workflow scene replaces it
        # in ``configure_env_cfg``; RL training and its videos keep this one.
        return NoTask().get_viewer_cfg()

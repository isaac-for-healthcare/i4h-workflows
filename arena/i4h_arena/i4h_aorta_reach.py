# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Bounded, kitless Arena feasibility task over i4h's unchanged aorta physics.

The physics is an i4h component (``i4h_isaaclab.component``): ``make_cfg`` declares it from the ``aorta_static``
case and ``AortaReachEnv`` builds through its handle, which rebinds the scene's catheter asset after every solver
build. The scene has no rigid bodies, so i4h steps alone (``configure(rigid=False)``)."""

from __future__ import annotations

import torch
import warp as wp
from i4h_endoluminal.assessment import single_target_assessor
from i4h_endoluminal.bindings import CatheterBinding
from i4h_isaaclab.component import I4hComponent, I4hComponentCfg
from isaaclab.assets import AssetBase, AssetBaseCfg
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    EventTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.sim import SimulationContext
from isaaclab.utils import configclass
from isaaclab_arena.assets.asset import Asset as ArenaAsset
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.arena_env_builder_cfg import ArenaEnvBuilderCfg
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
from isaaclab_arena.progress_tracking.completion_criteria import CompletionCriteria
from isaaclab_arena.scene.scene import Scene
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.physics_backend import PhysicsBackend
from isaaclab_newton.physics import NewtonCfg


class PhysicsAsset(AssetBase):
    """Live particle bindings; scene extras never own episode reset or stepping."""

    def __init__(self, cfg):
        self.sim = None
        self.assessment = None
        self._remaining_steps = 0
        super().__init__(cfg)

    def _initialize_impl(self):
        pass

    @property
    def num_instances(self):
        return 0 if self.sim is None else self.sim.scene.envs

    @property
    def data(self):
        return self

    def reset(self, env_ids=None):
        pass  # The explicit workflow reset event below owns complete reset.

    def update(self, dt):
        # Lab may update the scene once per physics step or once per decimated
        # action. Collect the full action in either case before replaying evidence.
        if self._remaining_steps:
            remaining = self._remaining_steps - round(dt / self.sim.dt)
            if remaining < 0:
                raise RuntimeError("scene update exceeded configured evidence capacity")
            if remaining == 0:
                self.assessment.consume(self.evidence, self.sim.dt, validate=self.cfg.validate_evidence)
            self._remaining_steps = remaining

    def write_data_to_sim(self):
        pass

    def bind(self, sim):
        self.sim = sim
        self._remaining_steps = 0
        assert self.sim.procedure is None
        scene = self.sim.scene
        self.owner = sim.physics
        self.physics = CatheterBinding(sim)
        self.evidence = self.physics.configure_tip_evidence(self.physics, self.env.cfg.decimation)
        self.q = wp.to_torch(self.physics.q)
        self.qd = wp.to_torch(self.physics.qd)
        self.flags = wp.to_torch(self.physics.flags)
        self.commands = wp.to_torch(self.physics.commands)
        self.targets = wp.to_torch(self.physics.targets)
        self.assessment = single_target_assessor(
            self.physics,
            target=self.spec["target"]["position"],
            radius=self.spec["radius"],
            hold=self.spec["hold"],
            timeout=self.spec["timeout"],
        )
        self.rows = wp.to_torch(self.assessment.rows)
        self.reward = wp.to_torch(self.assessment.action_reward)
        self.history = wp.to_torch(self.assessment.history)
        self.progress = wp.to_torch(self.assessment.progress)
        self.terminal_rows = torch.zeros_like(self.rows)
        self.terminal_history = torch.full_like(self.history, -1)
        self.terminal_progress = torch.zeros_like(self.progress)
        self.new_terminal = torch.zeros(scene.envs, device=self.q.device, dtype=torch.bool)
        self.recorded = torch.zeros_like(self.new_terminal)
        self._stream = torch.cuda.ExternalStream(wp.get_stream(scene.device).cuda_stream, device=self.q.device)
        self._in_event = torch.cuda.Event()
        self._out_event = torch.cuda.Event()
        self.to_torch()
        if hasattr(self.env, "action_manager"):
            self.env.action_manager.get_term("catheter").reset()

    def to_warp(self):
        self._in_event.record(torch.cuda.current_stream(self.q.device))
        self._stream.wait_event(self._in_event)

    def to_torch(self):
        self._out_event.record(self._stream)
        torch.cuda.current_stream(self.q.device).wait_event(self._out_event)


@configclass
class PhysicsAssetCfg(AssetBaseCfg):
    class_type: type = PhysicsAsset
    prim_path: str = "/World/I4hParticles"
    validate_evidence: bool = False  # Reads device counters and synchronizes at consumption.


class AortaAsset(ArenaAsset):
    def __init__(self):
        super().__init__("catheter", tags=["medical"])

    def get_object_cfg(self):
        return self.name, PhysicsAssetCfg()

    def get_event_cfg(self):
        return self.name, None


def reset_episode(env, env_ids):
    asset = env.scene["catheter"]
    ids = slice(None) if env_ids is None else env_ids
    asset.to_torch()
    selected = torch.zeros_like(asset.recorded)
    selected[ids] = True
    fresh = selected & (asset.rows[:, 11] != 0) & ~asset.recorded
    asset.terminal_rows.copy_(torch.where(fresh[:, None], asset.rows, asset.terminal_rows))
    asset.terminal_history.copy_(torch.where(fresh[:, None, None], asset.history, asset.terminal_history))
    asset.terminal_progress.copy_(torch.where(fresh[:, None], asset.progress, asset.terminal_progress))
    asset.new_terminal.logical_or_(fresh)
    asset.recorded.logical_or_(fresh)
    env.extras["terminal_task"] = asset.terminal_rows.clone()
    env.extras["terminal_history"] = asset.terminal_history.clone()
    env.extras["terminal_progress"] = asset.terminal_progress.clone()
    env.extras["new_terminal"] = asset.new_terminal.clone()
    asset.to_warp()
    asset.owner.reset(env_ids)
    asset.assessment.reset(env_ids)
    env.action_manager.get_term("catheter").reset(env_ids)
    asset.recorded[ids] = False


class CatheterAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 3), device=env.device)
        self._processed = torch.zeros_like(self._raw)

    @property
    def action_dim(self):
        return 3

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._processed

    def process_actions(self, actions):
        if self._env.cfg.decimation != self._asset.evidence.capacity:
            raise ValueError("decimation is fixed; reconstruct the environment")
        self._asset._remaining_steps = self._asset.evidence.capacity
        self._raw.copy_(actions)
        self._processed.copy_(actions.clamp(-1, 1))
        self._asset.new_terminal.zero_()
        self._env.extras["new_terminal"] = self._asset.new_terminal.clone()
        self._asset.to_warp()
        self._asset.evidence.begin_action()
        self._asset.assessment.begin_action()

    def apply_actions(self):
        self._asset.commands.copy_(self._processed)
        self._asset.to_warp()

    def reset(self, env_ids=None):
        self._asset._remaining_steps = 0
        ids = slice(None) if env_ids is None else env_ids
        self._raw[ids] = 0
        self._processed[ids] = 0


@configclass
class CatheterActionCfg(ActionTermCfg):
    class_type: type = CatheterAction
    asset_name: str = "catheter"


@configclass
class ActionsCfg:
    catheter = CatheterActionCfg()


def positions(env):
    asset = env.scene["catheter"]
    asset.to_torch()
    return asset.q.flatten(1)


def velocities(env):
    asset = env.scene["catheter"]
    asset.to_torch()
    return asset.qd.flatten(1)


def validity(env):
    asset = env.scene["catheter"]
    asset.to_torch()
    return asset.flags


def targets(env):
    asset = env.scene["catheter"]
    asset.to_torch()
    return asset.targets


def task_rows(env):
    asset = env.scene["catheter"]
    asset.to_torch()
    return asset.rows


@configclass
class PolicyObservations(ObservationGroupCfg):
    q = ObservationTermCfg(func=positions)
    qd = ObservationTermCfg(func=velocities)
    flags = ObservationTermCfg(func=validity)
    control_targets = ObservationTermCfg(func=targets)
    enable_corruption: bool = False
    concatenate_terms: bool = True


@configclass
class TaskObservations(ObservationGroupCfg):
    assessment = ObservationTermCfg(func=task_rows)
    enable_corruption: bool = False
    concatenate_terms: bool = True


@configclass
class ObservationsCfg:
    policy = PolicyObservations()
    task = TaskObservations()


def completed(env):
    return task_rows(env)[:, 11] == 1


def invalid(env):
    return task_rows(env)[:, 11] == 3


def timed_out(env):
    return task_rows(env)[:, 11] == 2


def completion_reward(env):
    asset = env.scene["catheter"]
    asset.to_torch()
    return asset.reward / env.step_dt


class CatheterEmbodiment(EmbodimentBase):
    name = "i4h_catheter"

    def __init__(self):
        super().__init__()
        self.action_config = ActionsCfg()
        self.observation_config = ObservationsCfg()


class AortaReachTask(TaskBase):
    name = "i4h_aorta_reach"
    # Authoritative workflow policy; legacy YAML remains reference compatibility.
    spec = dict(
        type="reach_target",
        tool="catheter",
        target={"position": [9.69, 0.18, 0.38]},
        radius=0.1,
        hold=0.5,
        timeout=120.0,
    )

    def get_scene_cfg(self):
        return None

    def get_termination_cfg(self):
        return TaskTerminationCfg(
            timeout_s=None,
            success=[CompletionCriteria("assessment_completed", predicate_sequence=[completed])],
            failures={"invalid_target": TerminationTermCfg(func=invalid)},
        )

    def get_events_cfg(self):
        @configclass
        class Events:
            reset = EventTermCfg(func=reset_episode, mode="reset")

        return Events()

    def get_rewards_cfg(self):
        @configclass
        class Rewards:
            completion = RewardTermCfg(func=completion_reward, weight=1.0)

        return Rewards()

    def get_mimic_env_cfg(self, arm_mode):
        raise NotImplementedError("Mimic is outside the bounded feasibility task")

    def get_metrics(self):
        return []


CASE = "aorta_static"  # i4h_endoluminal's catheter in the static aorta


def make_cfg(num_envs=1, *, captured=True, decimation=1, validate_evidence=False):
    if isinstance(decimation, bool) or not isinstance(decimation, int) or decimation < 1:
        raise ValueError("decimation must be a positive integer")
    physics = I4hComponentCfg.from_case(CASE)

    def configure(cfg):
        cfg.sim.dt = physics.sim["dt"]
        cfg.sim.render_interval = 1
        cfg.decimation = decimation
        cfg.scene.catheter.validate_evidence = validate_evidence
        cfg.scene.replicate_physics = True  # The public world hook supplies local particles.
        cfg.sim.physics = NewtonCfg(use_cuda_graph=captured)
        I4hComponent(physics).configure(cfg.sim, rigid=False)  # no rigid bodies in this scene: i4h steps alone
        cfg.compute_final_obs = True
        cfg.apply_rtx_global_settings = False
        return cfg

    description = IsaacLabArenaEnvironment(
        name="I4h-AortaReach-Manager",
        scene=Scene([AortaAsset()]),
        embodiment=CatheterEmbodiment(),
        task=AortaReachTask(),
        default_physics_backend=PhysicsBackend.NEWTON,
        env_cfg_callback=configure,
    )
    cfg, kwargs = ArenaEnvBuilder(
        description, ArenaEnvBuilderCfg(num_envs=num_envs, solve_relations=False)
    ).compose_manager_cfg()
    # Use the assessor's capped deadline and precedence, rather than Lab's control-step timeout.
    cfg.terminations.assessment_timeout = TerminationTermCfg(func=timed_out, time_out=True)
    cfg.recorders = None
    cfg.episode_recorders = None
    return cfg, kwargs


def make_env(num_envs=1, *, captured=True, decimation=1, validate_evidence=False):
    cfg, kwargs = make_cfg(num_envs, captured=captured, decimation=decimation, validate_evidence=validate_evidence)
    return AortaReachEnv(cfg, **kwargs)


class AortaReachEnv(IsaacLabArenaManagerBasedRLEnv):
    def __init__(self, cfg, **kwargs):
        if isinstance(cfg.decimation, bool) or not isinstance(cfg.decimation, int) or cfg.decimation < 1:
            raise ValueError("decimation must be a positive integer")
        self.component = I4hComponent.from_solver_cfg(cfg.sim.physics.solver_cfg)
        self.component.on_rebuild(lambda component: self._bind_physics(component.sim))  # after every solver build
        try:
            with self.component.construction(device=cfg.sim.device, envs=cfg.scene.num_envs):
                super().__init__(cfg, **kwargs)
        except BaseException:
            if SimulationContext.instance() is not None:
                SimulationContext.clear_instance()
            raise

    def _bind_physics(self, sim):
        asset = self.scene["catheter"]
        asset.env = self
        asset.spec = AortaReachTask.spec
        asset.bind(sim)
        if not hasattr(self, "_base_scene_update"):
            # Pinned Lab does not update AssetBase extras and does not instantiate
            # scene.cfg.class_type. Forward this instance's scene lifecycle only.
            self._base_scene_update = self.scene.update
            self.scene.update = self._update_scene

    def _update_scene(self, dt):
        self._base_scene_update(dt)
        self.scene["catheter"].update(dt)

    def _reset_idx(self, env_ids):
        # Arena's episode counters still expect indices; current Lab uses slices
        # for full manual reset, including with both recording managers disabled.
        if isinstance(env_ids, slice):
            env_ids = torch.arange(self.num_envs, device=self.device)[env_ids]
        super()._reset_idx(env_ids)

    def close(self):
        self.component.detach()
        super().close()


def close_env(env):
    env.close()
    if SimulationContext.instance() is not None:
        SimulationContext.clear_instance()

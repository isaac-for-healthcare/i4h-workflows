# SPDX-License-Identifier: Apache-2.0
"""Virtual grasper and Arena TaskBase hosted by the pinned Numerical Sim bridge."""

from __future__ import annotations

import copy
import os

import torch
from i4h_isaaclab.manager import I4H_ROOT, I4hHostMixin, configure_physics
from isaaclab.managers import (
    ActionTerm,
    ActionTermCfg,
    ObservationGroupCfg,
    ObservationTermCfg,
    RewardTermCfg,
    TerminationTermCfg,
)
from isaaclab.utils.configclass import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
from isaaclab_arena.scene.scene import Scene as ArenaScene
from isaaclab_arena.tasks.task_base import TaskBase
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.physics_backend import PhysicsBackend

from i4h_arena.assets.config_asset import ConfigAsset

from .runtime import LiverRuntime


def observation(env):
    return env.liver.observation()


def reward(env):
    return env.liver.control_reward()


def success(env):
    env.i4h.sync_views()
    return (env.liver.objective.s[:, 0] == 7) & (env.liver.views["target"] >= 0)


def failed(env):
    env.i4h.sync_views()
    return env.liver.objective.s[:, 0] == 8


class LiverAction(ActionTerm):
    def __init__(self, cfg, env):
        super().__init__(cfg, env)
        self._raw = torch.zeros((env.num_envs, 4), device=env.device)

    @property
    def action_dim(self):
        return 4

    @property
    def raw_actions(self):
        return self._raw

    @property
    def processed_actions(self):
        return self._env.liver.previous_action

    def process_actions(self, actions):
        self._raw.copy_(actions)
        self._env.liver.process(actions)

    def apply_actions(self):
        self._env.liver.apply()

    def reset(self, env_ids=None):
        self._raw[slice(None) if env_ids is None else env_ids] = 0


@configclass
class LiverActionCfg(ActionTermCfg):
    class_type: type = LiverAction
    asset_name: str = "i4h"


@configclass
class Actions:
    grasper: LiverActionCfg = LiverActionCfg()


@configclass
class PolicyObservation(ObservationGroupCfg):
    state: ObservationTermCfg = ObservationTermCfg(func=observation)
    enable_corruption: bool = False
    concatenate_terms: bool = True


@configclass
class Observations:
    policy: PolicyObservation = PolicyObservation()


@configclass
class Rewards:
    objective: RewardTermCfg = RewardTermCfg(func=reward, weight=1.0)


class LiverTask(TaskBase):
    def __init__(self):
        super().__init__(
            episode_length_s=20.0,
            task_description="Reach, acquire intended liver patch, retain, retract 20 mm, hold, release and relax.",
        )

    def get_scene_cfg(self):
        return None

    def get_events_cfg(self):
        return None

    def get_metrics(self):
        return []

    def get_mimic_env_cfg(self, arm_mode):
        return None

    def get_observation_cfg(self):
        return Observations()

    def get_rewards_cfg(self):
        return Rewards()

    def get_termination_cfg(self):
        # The 20 s deadline is an Arena task failure, never a bootstrapped truncation.
        return TaskTerminationCfg(timeout_s=None, failures={"liver_failure": TerminationTermCfg(func=failed)})


class VirtualGrasper(EmbodimentBase):
    registry_name = "virtual_grasper"

    def __init__(self):
        super().__init__()
        self.action_config = Actions()


class LiverEnv(I4hHostMixin, IsaacLabArenaManagerBasedRLEnv):
    def __init__(self, *args, **kwargs):
        self._liver_display = None
        super().__init__(*args, **kwargs)
        if self.sim.has_gui:
            from i4h_arena.assets.liver_reach_retract import LiverDisplay

            self._liver_display = LiverDisplay(self)

    def _bind_i4h(self, component):
        super()._bind_i4h(component)
        self.liver = LiverRuntime(self, component)

    def i4h_restore_ids(self, env_ids):
        if os.environ.get("I4H_LIVER_PREPARE") == "1" or getattr(self, "liver_ordinary_reset", False):
            return env_ids[:0]
        if not self.liver.snapshot_path.is_file():
            raise FileNotFoundError(f"Generate hosted offline snapshots first: {self.liver.snapshot_path}")
        return env_ids

    def i4h_after_reset(self, env_ids):
        self.liver.probe.clear(env_ids.tolist())
        self.liver.objective.reset(env_ids)
        self.liver.objective.s[env_ids, 0] = 7  # inactive until a target is initialized
        self.i4h.sync_views()
        self.liver.held[env_ids] = self.liver.physical[env_ids]
        self.liver.previous_action[env_ids] = 0
        self.i4h.order_inputs()

    def i4h_save_terminal(self, env_ids):
        rows = self.liver.terminal(env_ids)
        if rows:
            self.extras["terminal_results"] = rows
            import logging

            for row in rows:
                logging.getLogger("i4h_arena.liver").info("terminal %s", row)

    def i4h_restore_state(self, env_ids):
        specs = [self.liver.next_specs[int(e)] for e in env_ids] if self.liver.next_specs else None
        self.liver.restore(env_ids, specs)

    def step(self, action):
        # Prevent a terminal payload from being mistaken for a later episode.
        self.extras.pop("terminal_results", None)
        result = super().step(action)
        if self._liver_display is not None:
            self._liver_display.update()
        return result

    def capture_complete_state(self):
        return self.liver.capture_state()

    def restore_complete_state(self, state, env_ids=None):
        ids = torch.arange(self.num_envs, device=self.device) if env_ids is None else env_ids
        self.liver.restore_complete_state(state, ids)


class LiverBuilder(ArenaEnvBuilder):
    def get_entry_point(self):
        return LiverEnv


def make_environment(snapshot_dir, objective="full_cycle"):
    def configure(cfg):
        cfg.sim.dt = 1 / 60
        cfg.decimation = 2
        cfg.sim.render_interval = 2
        configure_physics(cfg.sim, "liver_pinned_grasp_1x", use_cuda_graph=True)
        cfg.compute_final_obs = True
        # The reducer already owns all objective state. A second progress tracker
        # would introduce another clock and inject a training recorder.
        cfg.terminations.success = TerminationTermCfg(func=success)
        cfg.task_progress = None
        cfg.liver_snapshot_dir = str(snapshot_dir)
        cfg.liver_objective = objective
        cfg.viewer.eye = (0.45, 0.45, -0.05)
        cfg.viewer.lookat = (0.04, 0.08, -0.51)
        return cfg

    return IsaacLabArenaEnvironment(
        name="liver_reach_retract",
        scene=ArenaScene(assets=[ConfigAsset("i4h", copy.deepcopy(I4H_ROOT))]),
        embodiment=VirtualGrasper(),
        task=LiverTask(),
        default_physics_backend=PhysicsBackend.NEWTON,
        env_cfg_callback=configure,
        rl_framework_entry_point="rsl_rl_cfg_entry_point",
        rl_policy_cfg="i4h_arena.agents.rsl_rl:ProfiledRslRlRunnerCfg",
    )

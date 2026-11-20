# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena composition over the pinned Numerical Sim static-aorta component."""

from __future__ import annotations

import copy

from i4h_isaaclab.manager import I4H_ROOT, I4hActionTermCfg, I4hHostMixin, configure_physics
from isaaclab.managers import RecorderTerm, RecorderTermCfg
from isaaclab.utils import configclass
from isaaclab_arena.embodiments.embodiment_base import EmbodimentBase
from isaaclab_arena.environments.arena_env_builder import ArenaEnvBuilder
from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
from isaaclab_arena.environments.isaaclab_arena_manager_based_env import IsaacLabArenaManagerBasedRLEnv
from isaaclab_arena.scene.scene import Scene as ArenaScene
from isaaclab_arena.tasks.no_task import NoTask
from isaaclab_arena.tasks.task_termination_cfg import TaskTerminationCfg
from isaaclab_arena.utils.physics_backend import PhysicsBackend

from i4h_arena.assets.config_asset import ConfigAsset
from i4h_arena.sensors.fluoroscopy import FluoroscopySensorCfg

# Fixed orthographic projection along Z. The rectangular detector field is
# intentionally retained on a square image: 45.1 mm/px in X, 15.7 mm/px in Y.
WORLD_BOUNDS_M = (-0.5, 11.0, -2.0, 2.0)


@configclass
class ActionsCfg:
    catheter = I4hActionTermCfg(mode="normalized")


class AortaDiagnosticsRecorder(RecorderTerm):
    def record_pre_step(self):
        import torch

        from i4h_arena.scenes.endoluminal_aorta_fluoroscopy import AortaStateProvider

        provider = AortaStateProvider(self._env)
        points = provider.points()
        return "diagnostics", {
            "root_world_m": torch.from_numpy(points[:, 0]),
            "tip_world_m": torch.from_numpy(points[:, -1]),
            "command_targets": torch.from_numpy(provider.targets()),
        }


@configclass
class AortaRecorderTermsCfg:
    record_aorta_diagnostics: RecorderTermCfg = RecorderTermCfg(class_type=AortaDiagnosticsRecorder)


def aorta_recorders():
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
    from isaaclab_arena.utils.configclass import combine_configclass_instances

    from i4h_arena.embodiments.recorders import action_state_recorders
    from i4h_arena.medical.catheter_recording import FluoroscopyFrameRecorder

    @configclass
    class FrameTerms:
        record_fluoroscopy_frames: RecorderTermCfg = RecorderTermCfg(class_type=FluoroscopyFrameRecorder)

    base = action_state_recorders()
    base.record_pre_step_flat_policy_observations = None
    return combine_configclass_instances(
        "AortaRecorderCfg", base, AortaRecorderTermsCfg(), FrameTerms(), bases=(RecorderManagerBaseCfg,)
    )


class NumericalCatheterEmbodiment(EmbodimentBase):
    registry_name = "numerical_catheter"

    def __init__(self):
        super().__init__()
        self.action_config = ActionsCfg()

    def get_recorder_term_cfg(self, record_trajectories=False):
        return aorta_recorders()


class AortaDiagnosticTask(NoTask):
    def get_termination_cfg(self):
        return TaskTerminationCfg(timeout_s=self.get_episode_length_s())


class AortaFluoroscopyEnv(I4hHostMixin, IsaacLabArenaManagerBasedRLEnv):
    """Use the component's construction, stream synchronization and reset order."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        from i4h_arena.scenes.endoluminal_aorta_fluoroscopy import AortaStateProvider

        self.scene["fluoroscopy"].bind_catheter_provider(AortaStateProvider(self))


class AortaArenaBuilder(ArenaEnvBuilder):
    def get_entry_point(self):
        # Keep the registered Arena factory, including its runtime kwargs.
        return AortaFluoroscopyEnv


def make_environment():
    def configure(cfg):
        cfg.sim.dt = 1 / 60
        cfg.decimation = 2
        cfg.sim.render_interval = 2
        configure_physics(cfg.sim, "aorta_static", use_cuda_graph=True)
        cfg.compute_final_obs = True
        cfg.viewer.eye = (5.0, -12.0, 10.0)
        cfg.viewer.lookat = (5.0, 0.0, 1.0)
        return cfg

    arena_env = IsaacLabArenaEnvironment(
        name="endoluminal_aorta_fluoroscopy",
        scene=ArenaScene(
            assets=[
                ConfigAsset("i4h", copy.deepcopy(I4H_ROOT)),
                ConfigAsset(
                    "fluoroscopy",
                    FluoroscopySensorCfg(
                        prim_path="{ENV_REGEX_NS}",
                        width=256,
                        height=256,
                        backend="synthetic",
                        world_bounds_m=WORLD_BOUNDS_M,
                        demo_catheter=False,
                    ),
                ),
            ]
        ),
        embodiment=NumericalCatheterEmbodiment(),
        task=AortaDiagnosticTask(episode_length_s=600 / 30),
        default_physics_backend=PhysicsBackend.NEWTON,
        env_cfg_callback=configure,
    )
    arena_env.env_entry_point = AortaFluoroscopyEnv
    return arena_env

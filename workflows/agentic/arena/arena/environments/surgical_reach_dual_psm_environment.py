# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import Any

from arena.environments.core.surgical_base import SurgicalStateMachineEnvironmentBase


class SurgicalReachDualPsmEnvironment(SurgicalStateMachineEnvironmentBase):
    name: str = "surgical_reach_dual_psm"
    state_machine_module: str = "arena.statemachine.surgical_reach_dual_psm"

    def get_env(self, args: argparse.Namespace) -> Any:
        from arena.assets.surgical_scenes import make_surgical_scene_assets
        from arena.embodiments.psm import DualPsmEmbodiment
        from arena.tasks.surgical_target import SurgicalReachCommandTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        embodiment = DualPsmEmbodiment(
            enable_cameras=args.enable_cameras,
            action_device=getattr(args, "action_device", "joint_position"),
            sim_decimation=2,
            render_interval=2,
            enable_material_randomization=False,
        )
        scene = Scene(assets=make_surgical_scene_assets("reach_dual_psm"))
        return IsaacLabArenaEnvironment(
            name=self.name, embodiment=embodiment, scene=scene, task=SurgicalReachCommandTask.dual_psm()
        )

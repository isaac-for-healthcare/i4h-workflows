# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
from typing import Any

from arena.environments.core.surgical_base import SurgicalStateMachineEnvironmentBase


class SurgicalLiftBlockEnvironment(SurgicalStateMachineEnvironmentBase):
    name: str = "surgical_lift_block"
    state_machine_module: str = "arena.statemachine.surgical_lift_block"

    def get_env(self, args: argparse.Namespace) -> Any:
        from arena.assets.surgical_scenes import make_surgical_scene_assets
        from arena.embodiments.psm import PSMEmbodiment
        from arena.tasks.surgical_target import SurgicalLiftTask
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene

        embodiment = PSMEmbodiment(
            enable_cameras=args.enable_cameras,
            action_device=getattr(args, "action_device", "joint_position"),
            sim_dt=1.0 / 200.0,
            sim_decimation=4,
            render_interval=4,
            gripper_close=0.1,
            enable_material_randomization=False,
        )
        scene = Scene(assets=make_surgical_scene_assets("lift_block"))
        task = SurgicalLiftTask(task_description="Lift the peg-transfer block with the dVRK PSM.", organs=False)
        return IsaacLabArenaEnvironment(name=self.name, embodiment=embodiment, scene=scene, task=task)

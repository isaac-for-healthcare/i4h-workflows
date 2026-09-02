# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Franka ultrasound probe reaching a randomized phantom target."""

from __future__ import annotations

from typing import Any

from i4h_arena.scenes.base import Scene


class UltrasoundProbeReachScene(Scene):
    name = "ultrasound_probe_reach"

    def register_assets(self) -> None:
        import i4h_arena.assets.ultrasound_probe_reach  # noqa: F401

    def build(self) -> Any:
        from isaaclab_arena.environments.isaaclab_arena_environment import IsaacLabArenaEnvironment
        from isaaclab_arena.scene.scene import Scene as ArenaScene

        from i4h_arena.assets.ultrasound_probe_reach import make_assets
        from i4h_arena.embodiments.franka import (
            FRANKA_ULTRASOUND_READY_JOINT_POS,
            FrankaUltrasoundEmbodiment,
            FrankaUltrasoundRLEmbodiment,
        )
        from i4h_arena.envcfg.ultrasound_probe_reach import UltrasoundProbeReachTask

        rl_observations = bool(
            getattr(self.args, "rl_observations", False) or getattr(self.args, "mode", "") == "policy"
        )
        embodiment = (
            FrankaUltrasoundRLEmbodiment()
            if rl_observations
            else FrankaUltrasoundEmbodiment(enable_cameras=not bool(getattr(self.args, "no_cameras", False)))
        )
        if not rl_observations:
            embodiment.scene_config.robot.init_state.joint_pos = {
                f"panda_joint{index}": value for index, value in enumerate(FRANKA_ULTRASOUND_READY_JOINT_POS, start=1)
            }

        return IsaacLabArenaEnvironment(
            name=self.name,
            embodiment=embodiment,
            scene=ArenaScene(assets=make_assets()),
            task=UltrasoundProbeReachTask(
                rl_training_mode=bool(getattr(self.args, "rl_training_mode", False)),
                episode_length_s=5.0,
            ),
            rl_framework_entry_point="rsl_rl_cfg_entry_point",
            rl_policy_cfg="i4h_arena.agents.rsl_rl:ProfiledRslRlRunnerCfg",
        )

    def tcp_body(self) -> str | None:
        return "TCP"

    def tcp_sensors(self) -> dict[str, str]:
        return {"robot": "ee_frame"}

    def camera_aliases(self) -> dict[str, str]:
        return {"room": "room_camera", "wrist": "wrist_camera"}

    def relative_ee(self) -> bool:
        return True

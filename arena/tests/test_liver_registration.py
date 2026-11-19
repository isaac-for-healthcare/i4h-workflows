# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import sys
from types import ModuleType, SimpleNamespace

import i4h_arena.scenes.liver_reach_retract as scene_module
from i4h_arena.scenes.liver_reach_retract import LiverReachRetractScene


def test_transform_reaches_registered_custom_host_and_keeps_runtime_kwargs(monkeypatch):
    recorded = []
    runtime_kwargs = {"variation_recorder": object(), "variation_replay_scheduler": object()}
    cfg = SimpleNamespace(recorders=object())

    class CustomBuilder:
        def __init__(self, arena, options):
            pass

        def build_env_cfg(self):
            return cfg, runtime_kwargs

        def build_registered(self, env_cfg, env_kwargs):
            assert env_cfg.recorders is None
            assert env_kwargs is runtime_kwargs
            recorded.append((env_cfg, env_kwargs))
            return "custom-liver-host", env_cfg, env_kwargs

    module = ModuleType("i4h_arena.envcfg.liver_reach_retract.environment")
    module.LiverBuilder = CustomBuilder
    monkeypatch.setitem(sys.modules, module.__name__, module)
    monkeypatch.setattr(scene_module, "arena_builder_cfg", lambda args: args)
    scene = LiverReachRetractScene.__new__(LiverReachRetractScene)
    scene.args = SimpleNamespace()
    scene.arena_environment = lambda: object()
    scene.configure_env_cfg = lambda cfg: None

    def disable(cfg):
        cfg.recorders = None
        return cfg

    gym_id, registered = scene.gym_spec(config_transform=disable)
    assert gym_id == "custom-liver-host" and registered is cfg and len(recorded) == 1

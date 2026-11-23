# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned Arena video compatibility registers Lab's recorder in its normal lifecycle."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest


@pytest.mark.parametrize("native", [True, False])
def test_viewport_recorder_selects_native_lifecycle_or_legacy_gym(monkeypatch, tmp_path, native):
    root = Path(__file__).resolve().parents[2]
    source = root / "third_party/IsaacLab-Arena-196680c/isaaclab_arena/video/video_recording.py"
    if not source.is_file():
        pytest.skip("pinned Arena source unavailable")
    spec = importlib.util.spec_from_file_location("viewport_recording_under_test", source)
    module = importlib.util.module_from_spec(spec)
    monkeypatch.setitem(sys.modules, spec.name, module)
    spec.loader.exec_module(module)
    owner = SimpleNamespace(max_episode_length=600)
    if native:
        owner.video_recorders = []
    env = SimpleNamespace(unwrapped=owner)
    registrations = []

    class Recorder:
        def __init__(self, cfg, owner):
            self.cfg = cfg
            self.owner = owner

    def install(name, **attrs):
        fake = ModuleType(name)
        fake.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, fake)

    install("isaaclab_arena.utils.env_step_timer", EnvStepTimerWrapper=lambda env, **kw: env)
    install("isaaclab.envs.utils.video_recorder", VideoRecorder=Recorder)
    install("isaaclab.envs.utils.video_recorder_cfg", VideoRecorderCfg=lambda **kw: SimpleNamespace(**kw))
    import gymnasium.wrappers

    monkeypatch.setattr(gymnasium.wrappers, "RecordVideo", lambda env, **kw: registrations.append(kw) or env)
    cfg = module.VideoRecordingCfg(record_viewport_video=True, video_base_dir=str(tmp_path))
    assert module.wrap_env_for_video(env, cfg, None, 1) is env
    if native:
        assert registrations == []
        (recorder,) = owner.video_recorders
        assert recorder.owner is owner
        assert recorder.cfg.source == "visualizer"
        assert recorder.cfg.video_length == 600
        assert recorder.cfg.output_dir == str(tmp_path)
    else:
        assert registrations[0]["video_length"] == 600
        assert registrations[0]["step_trigger"](0)
        assert not registrations[0]["step_trigger"](1)

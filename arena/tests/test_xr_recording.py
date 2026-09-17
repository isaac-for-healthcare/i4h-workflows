# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise operator controls through the shared runner and real HDF5 writer."""

from collections import deque
from types import SimpleNamespace

import h5py
import numpy as np

from i4h_arena.adapters.actuation import ArenaActuation
from i4h_arena.recording.hdf5 import EpisodeRecorder
from i4h_arena.runner import SimulationRunner
from i4h_common.episode import read_segments
from i4h_engine.loader import resolve_workflow
from i4h_engine.registry import default_registry
from i4h_tasks.basic.testing.fake_scene import FakeScene


def test_discarded_and_waiting_frames_do_not_leak_into_saved_demo(tmp_path, monkeypatch):
    samples = deque(
        [
            (None, {"S"}),  # Cannot save before B.
            (None, {"B"}),
            (np.ones(38), set()),
            (None, {"R"}),  # Discard the first physical step.
            (None, {"S"}),
            (None, {"B"}),
            (np.zeros(38), set()),
            (None, {"S"}),
        ]
    )
    source = SimpleNamespace(open=lambda: None, close=lambda: None, reset=lambda: None, read=samples.popleft)
    monkeypatch.setattr("i4h_tasks.teleop.xr.XRInput", lambda *a, **kw: source)
    monkeypatch.setattr(
        "i4h_arena.runner._kit_settings_manager",
        lambda: SimpleNamespace(get=lambda *a: True, set_bool=lambda *a: None),
    )
    view = FakeScene(dof=53)
    view.invalidate = lambda: None
    view.teleop_config = lambda: object()
    counts = {"steps": 0, "resets": 0}

    def reset():
        counts["resets"] += 1
        view.joint_pos.fill(0)

    def step(action):
        counts["steps"] += 1
        view.joint_pos.fill(float(action[0, 0]))

    env = SimpleNamespace(reset=reset, step=step, close=lambda: None, step_dt=1 / 30)
    env.unwrapped = SimpleNamespace(sim=SimpleNamespace(render=lambda: None))
    workflow = resolve_workflow("spread_tablecloth_g1", "teleop")
    scene = SimpleNamespace(
        spec=default_registry().scene(workflow.scene),
        make_view=lambda env: view,
        make_actuation=lambda env, view: ArenaActuation(num_envs=1, action_dim=38, action_space="bimanual_pose"),
        on_reset=lambda env, view: None,
    )
    path = tmp_path / "demo.hdf5"
    recorder = EpisodeRecorder(path, workflow=workflow)
    try:
        summary = SimulationRunner(
            scene=scene,
            workflow=workflow,
            env=env,
            app=SimpleNamespace(is_running=lambda: bool(samples), update=lambda: None),
            recorder=recorder,
        ).run()
    finally:
        recorder.close()
    assert summary.complete
    assert counts == {"steps": 2, "resets": 4}
    with h5py.File(path) as f:
        assert list(f["data"]) == ["demo_0"]
        demo = f["data/demo_0"]
        assert demo["actions"].shape == (1, 38)
        assert demo["obs/joint_pos"].shape == (1, 53)
        assert not demo["actions"][()].any()
        assert demo.attrs["success"]
        (segment,) = read_segments(demo)
        assert (segment.task_id, segment.start, segment.end) == ("teleop/xr", 0, 1)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_common.bus.messages import ObsFrame
from i4h_common.server import Session

# isort: split
from i4h_tasks.gr00t_n16.server import Gr00tN16Server


def test_load_uses_task_modality_config(monkeypatch) -> None:
    loaded = []
    policy_module = SimpleNamespace(G1LocomanipClosedloopPolicy=lambda **kwargs: SimpleNamespace(reset=lambda: None))
    monkeypatch.setitem(
        sys.modules,
        "i4h_tasks.gr00t_n16.locomanip.infer.closedloop_policy",
        policy_module,
    )
    monkeypatch.setattr("i4h_tasks.gr00t_n16._finetune._load_modality_config", loaded.append)
    monkeypatch.setattr(
        Gr00tN16Server,
        "_declaration",
        lambda _self, _task_id: {"train": {"modality_config_path": "tasks/gr00t_n16/custom_config.py"}},
    )
    session = Session(
        task_uid="task-1",
        task_id="gr00t_n16/test_task",
        run_id="run-1",
        episode_index=0,
        prompt="walk to the table",
        checkpoint="checkpoint",
        model={"repo": "base"},
    )
    server = object.__new__(Gr00tN16Server)
    server._policies = {}

    server.load(session)

    assert loaded[0].as_posix().endswith("tasks/gr00t_n16/custom_config.py")


def test_cache_key_separates_task_modality_contracts() -> None:
    server = object.__new__(Gr00tN16Server)
    session = Session(
        task_uid="task-1",
        task_id="gr00t_n16/task_a",
        run_id="run-1",
        episode_index=0,
        prompt="walk to the table",
        checkpoint="checkpoint",
        model={"repo": "base"},
    )

    first = server._cache_key(session)
    session.task_id = "gr00t_n16/task_b"

    assert server._cache_key(session) != first


def test_infer_publishes_the_native_action_chunk(monkeypatch) -> None:
    class Tensor:
        def __init__(self, data) -> None:
            self.data = np.asarray(data)

        @property
        def shape(self):
            return self.data.shape

        def __getitem__(self, item):
            return Tensor(self.data[item])

        def detach(self):
            return self

        def cpu(self):
            return self

        def numpy(self):
            return self.data

    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(as_tensor=lambda data, **_kwargs: Tensor(data)))
    expected = np.arange(16 * 50, dtype=np.float32).reshape(1, 16, 50)

    class Policy:
        def get_action_chunk(self, observation):
            assert observation["policy"]["robot_joint_pos"].shape == (1, 43)
            assert observation["camera_obs"]["head"].shape == (1, 2, 2, 3)
            return Tensor(expected)

    session = Session(
        task_uid="task-1",
        task_id="gr00t_n16/locomanip_push_cart",
        run_id="run-1",
        episode_index=0,
        prompt="push the cart",
        checkpoint="",
        model={"repo": "checkpoint", "action_horizon": 16},
    )
    server = object.__new__(Gr00tN16Server)
    server._policies = {server._cache_key(session): Policy()}
    pixels = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    frame = ObsFrame(
        task_uid=session.task_uid,
        state=[0.0] * 43,
        images={"head": pixels.tobytes()},
        image_shapes={"head": [2, 2, 3]},
    )

    actions = server.infer(session, frame)

    assert actions is not None
    assert actions.shape == (16, 50)
    np.testing.assert_array_equal(actions, expected[0])

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena runs record through ``run.sh --record``, never through IsaacLab's recorder."""

import sys
from types import ModuleType, SimpleNamespace

import pytest
from isaaclab.envs.mdp.recorders.recorders_cfg import ActionStateRecorderManagerCfg
from isaaclab.managers.recorder_manager import RecorderManager

from i4h_arena.app import AppContext


def test_arena_env_has_no_isaaclab_dataset_export(monkeypatch):
    # What so_arm, psm, star, franka, kuka_lbr and g1 contribute through
    # get_recorder_term_cfg(): every step exported to /tmp/isaaclab/logs.
    cfg = SimpleNamespace(recorders=ActionStateRecorderManagerCfg(), terminations=SimpleNamespace(time_out=object()))
    scene = SimpleNamespace(name="test", register_assets=lambda: None, gym_spec=lambda: ("test", cfg))
    built = []

    def make(_id, **kwargs):
        built.append(kwargs["cfg"])
        raise RuntimeError("stop at gym construction boundary")

    gymnasium = ModuleType("gymnasium")
    gymnasium.make = make
    monkeypatch.setitem(sys.modules, "gymnasium", gymnasium)
    with pytest.raises(RuntimeError, match="gym construction boundary"):
        AppContext(app=None, args=SimpleNamespace()).make_env(scene)

    assert built and built[0].recorders is None
    # IsaacLab treats a None cfg as no recorder terms and opens no dataset file.
    assert RecorderManager(built[0].recorders, env=None).active_terms == []

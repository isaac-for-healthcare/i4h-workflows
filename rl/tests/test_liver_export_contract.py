# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export provenance follows the installed source, not an obsolete integration constant."""

import importlib.metadata
import json
import os
import subprocess
import sys
from types import ModuleType, SimpleNamespace

import pytest

from i4h_rl.adapters.liver_reach_retract import export_contract


@pytest.fixture
def export_inputs(tmp_path, monkeypatch):
    numerical = tmp_path / "physics"
    numerical.mkdir()
    module = ModuleType("i4h_physics")
    module.__file__ = str(numerical / "src/i4h_physics/__init__.py")
    monkeypatch.setitem(sys.modules, "i4h_physics", module)
    for name in ("i4h_arena", "i4h_arena.medical", "i4h_arena.medical.liver"):
        monkeypatch.setitem(sys.modules, name, ModuleType(name))
    liver = sys.modules["i4h_arena.medical.liver"]
    liver.ENCODER, liver.REWARD_REVISION, liver.TASK_CONTRACT = "encoder", "reward", "task"
    origin = {"url": numerical.as_uri(), "dir_info": {"editable": True}}
    original = importlib.metadata.distribution
    monkeypatch.setattr(
        importlib.metadata,
        "distribution",
        lambda name: (
            SimpleNamespace(read_text=lambda _: json.dumps(origin)) if name == "i4h-physics" else original(name)
        ),
    )
    checkpoint, policy = tmp_path / "checkpoint.pt", tmp_path / "policy.pt"
    checkpoint.write_bytes(b"checkpoint")
    policy.write_bytes(b"policy")
    env = SimpleNamespace(liver=SimpleNamespace(objective=SimpleNamespace(objective="full_cycle")))
    actor = SimpleNamespace(distribution=SimpleNamespace(std_type="scalar"))
    return numerical, origin, (env, actor, checkpoint, policy)


def test_export_records_the_actual_installed_git_revision(export_inputs):
    numerical, _, inputs = export_inputs
    git_env = dict(
        os.environ,
        GIT_CONFIG_NOSYSTEM="1",
        GIT_CONFIG_GLOBAL=os.devnull,
        GIT_AUTHOR_NAME="i4h",
        GIT_AUTHOR_EMAIL="i4h@example.com",
        GIT_COMMITTER_NAME="i4h",
        GIT_COMMITTER_EMAIL="i4h@example.com",
    )

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=numerical, env=git_env, text=True).strip()

    git("init", "-q")
    revisions = []
    for message in ("first source", "next source"):
        git("commit", "--allow-empty", "-qm", message)
        revision = git("rev-parse", "HEAD")
        export_contract(*inputs)
        contract = json.loads(inputs[-1].with_name("contract.json").read_text())
        assert contract["numerical_revision"] == revision
        assert contract["objective"] == "full_cycle" and len(contract["policy_sha256"]) == 64
        revisions.append(revision)
    assert revisions[0] != revisions[1]


def test_export_rejects_untraceable_wheel_source(export_inputs):
    _, origin, inputs = export_inputs
    origin["dir_info"]["editable"] = False
    with pytest.raises(RuntimeError, match="identifiable editable"):
        export_contract(*inputs)
    assert not inputs[-1].with_name("contract.json").exists()

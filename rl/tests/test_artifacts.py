# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run metadata never holds a credential, and the user's relative paths mean where the user typed them."""

from __future__ import annotations

from pathlib import Path

import pytest

from i4h_rl.artifacts import environment_overrides, resolve_input_path, resolve_output_path, resolve_run_dir


@pytest.mark.parametrize(
    "name",
    [
        "I4H_AGENT_API_KEY",
        "I4H_AGENT_VL_API_KEY",
        "I4H_HF_TOKEN",
        "I4H_SOME_SECRET",
        "I4H_DB_PASSWORD",
        "I4H_RL_SIM_AUTHKEY",
    ],
)
def test_a_credential_is_never_recorded_as_an_override(name) -> None:
    """NVBug 7028691: ``./train.sh rl`` wrote I4H_AGENT_API_KEY into run.json."""
    overrides = environment_overrides({name: "qa-canary", "I4H_CATHETER_SEGMENTS": "40"})

    assert overrides == {"I4H_CATHETER_SEGMENTS": "40"}


def test_the_launchers_own_variables_are_not_recorded() -> None:
    assert environment_overrides({"I4H_CALLER_CWD": "/somewhere", "I4H_WORKFLOWS": "/repo"}) == {}


def test_a_relative_run_dir_is_relative_to_the_caller(tmp_path, monkeypatch) -> None:
    """NVBug 7035808: train.sh put a relative --run-dir under the repository root; run.sh used the caller's."""
    root, caller = tmp_path / "repo", tmp_path / "caller"
    root.mkdir()
    caller.mkdir()
    monkeypatch.setenv("I4H_CALLER_CWD", str(caller))

    assert resolve_run_dir(root, "ultrasound_probe_reach", "rel_run") == caller / "rel_run"
    assert resolve_output_path(root, "export") == caller / "export"


def test_the_default_run_dir_stays_under_the_repository(tmp_path, monkeypatch) -> None:
    root, caller = tmp_path / "repo", tmp_path / "caller"
    monkeypatch.setenv("I4H_CALLER_CWD", str(caller))

    assert resolve_run_dir(root, "ultrasound_probe_reach", None).parent.parent == root / "runs"


def test_an_input_is_found_in_the_callers_directory_before_the_repository(tmp_path, monkeypatch) -> None:
    root, caller = tmp_path / "repo", tmp_path / "caller"
    for folder in (root, caller):
        folder.mkdir()
        (folder / "model.pt").write_text(folder.name)
    monkeypatch.setenv("I4H_CALLER_CWD", str(caller))

    assert resolve_input_path(root, "model.pt").read_text() == "caller"
    (caller / "model.pt").unlink()
    assert resolve_input_path(root, "model.pt").read_text() == "repo"


def test_without_a_caller_the_repository_root_is_the_base(tmp_path, monkeypatch) -> None:
    monkeypatch.delenv("I4H_CALLER_CWD", raising=False)

    assert resolve_output_path(tmp_path, "out") == Path(tmp_path / "out").resolve()

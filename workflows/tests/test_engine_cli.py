# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``i4h_engine.cli``: ``lint --all`` and ``show --format mermaid`` (NVBugs 7035809, 7042161)."""

from __future__ import annotations

import types

import pytest

from i4h_engine import cli
from i4h_engine.loader import WorkflowModule, load_workflow_module, resolve_workflow
from i4h_engine.render import to_mermaid

GOOD = "surgical_reach_psm"


def test_lint_all_reports_a_broken_module_and_lints_the_rest(monkeypatch, capsys) -> None:
    def load(name):
        if name == "broken_workflow":
            raise ModuleNotFoundError("No module named 'missing_dependency'")
        return load_workflow_module(name)

    monkeypatch.setattr(cli, "available_workflows", lambda: ["broken_workflow", GOOD])
    monkeypatch.setattr(cli, "load_workflow_module", load)

    code = cli.main(["lint", "--all"])

    out = capsys.readouterr().out
    assert code == 1
    assert "broken_workflow: !! ModuleNotFoundError: No module named 'missing_dependency'" in out
    assert f"{GOOD} [" in out


@pytest.mark.parametrize(
    ("export", "expected"),
    [
        (None, "must export WORKFLOW"),
        ("not a workflow", "exports WORKFLOW as str"),
    ],
)
def test_lint_all_reports_a_module_without_a_valid_workflow_export(monkeypatch, capsys, export, expected) -> None:
    broken = types.ModuleType("broken_workflow")
    if export is not None:
        broken.WORKFLOW = export

    def load(name):
        if name == "broken_workflow":
            return WorkflowModule(name=name, module=broken)
        return load_workflow_module(name)

    monkeypatch.setattr(cli, "available_workflows", lambda: ["broken_workflow", GOOD])
    monkeypatch.setattr(cli, "load_workflow_module", load)

    code = cli.main(["lint", "--all"])

    out = capsys.readouterr().out
    assert code == 1
    assert "broken_workflow: !! " in out
    assert expected in out
    assert f"{GOOD} [" in out


def test_lint_all_reports_a_mode_that_fails_to_resolve_and_lints_the_rest(monkeypatch, capsys) -> None:
    modes = sorted(load_workflow_module(GOOD).modes)
    bad_mode = modes[0]

    def resolve(name, mode):
        if mode == bad_mode:
            raise RuntimeError("builder failed")
        return resolve_workflow(name, mode)

    monkeypatch.setattr(cli, "available_workflows", lambda: [GOOD])
    monkeypatch.setattr(cli, "resolve_workflow", resolve)

    code = cli.main(["lint", "--all"])

    out = capsys.readouterr().out
    assert code == 1
    assert f"{GOOD} [{bad_mode}]: !! RuntimeError: builder failed" in out
    for mode in modes[1:]:
        assert f"{GOOD} [{resolve_workflow(GOOD, mode).mode}]" in out


def test_lint_all_passes_when_every_workflow_lints(monkeypatch, capsys) -> None:
    monkeypatch.setattr(cli, "available_workflows", lambda: [GOOD])

    assert cli.main(["lint", "--all"]) == 0
    assert "!!" not in capsys.readouterr().out


@pytest.mark.parametrize("mode", [None, "rule-based"])
def test_mermaid_is_bare_mermaid_source(mode) -> None:
    text = to_mermaid(resolve_workflow(GOOD, mode))
    lines = text.splitlines()

    assert lines[0] == "flowchart TD"
    assert not any(line.startswith("```") for line in lines)
    assert len(lines) > 1

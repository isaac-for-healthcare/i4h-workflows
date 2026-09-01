# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Path-resolution regression tests for the shared workflow launcher."""

from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

WORKFLOWS_ROOT = Path(__file__).resolve().parent.parent
RUN_SH = WORKFLOWS_ROOT / "run.sh"
STOP_SH = WORKFLOWS_ROOT / "stop.sh"
WORKFLOW = "scissor_pick_and_place"


def _dry_run(cwd: Path, *args: str) -> str:
    result = subprocess.run(
        [str(RUN_SH), WORKFLOW, "--dry-run", "--no-backend", *args],
        cwd=cwd,
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout


def _reported(output: str, label: str) -> Path:
    match = re.search(rf"^==> {re.escape(label)} (.+)$", output, re.MULTILINE)
    assert match is not None, output
    return Path(match.group(1))


def test_default_run_dir_and_bare_record_are_created_together() -> None:
    output = _dry_run(WORKFLOWS_ROOT, "--record")
    run_dir = _reported(output, "run dir")
    try:
        assert run_dir.parent == WORKFLOWS_ROOT / "runs" / WORKFLOW
        assert re.fullmatch(r"\d{8}_\d{6}(?:_\d{2})?", run_dir.name)
        assert run_dir.is_dir()
        assert _reported(output, "recording") == run_dir / "demos.hdf5"
        metadata = json.loads((run_dir / "run.json").read_text())
        assert metadata["run_dir"] == str(run_dir)
        assert metadata["recording"] == str(run_dir / "demos.hdf5")
        assert metadata["workflow"] == WORKFLOW
        assert (run_dir / "i4h_arena.log").read_text().splitlines()[0] == f"I4H_RUN_DIR={run_dir}"
    finally:
        shutil.rmtree(run_dir)


def test_explicit_relative_run_dir_is_relative_to_caller(tmp_path: Path) -> None:
    output = _dry_run(tmp_path, "--run-dir", "artifacts/session", "--record", "captures/demo.hdf5")

    run_dir = tmp_path / "artifacts" / "session"
    assert _reported(output, "run dir") == run_dir
    assert _reported(output, "recording") == run_dir / "captures" / "demo.hdf5"
    assert (run_dir / "captures").is_dir()
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["run_dir"] == str(run_dir)
    assert metadata["caller_cwd"] == str(tmp_path)


def test_absolute_run_and_record_paths_are_preserved(tmp_path: Path) -> None:
    run_dir = tmp_path / "selected-run"
    recording = tmp_path / "shared" / "demos.hdf5"
    output = _dry_run(tmp_path, "--run-dir", str(run_dir), "--record", str(recording))

    assert _reported(output, "run dir") == run_dir
    assert _reported(output, "recording") == recording
    assert run_dir.is_dir()
    assert recording.parent.is_dir()
    metadata = json.loads((run_dir / "run.json").read_text())
    assert metadata["recording"] == str(recording)


def test_patient_twin_input_is_relative_to_caller(tmp_path: Path) -> None:
    patient_twin = tmp_path / "inputs" / "patient_twin.yaml"
    patient_twin.parent.mkdir()
    patient_twin.touch()

    output = _dry_run(tmp_path, "--run-dir", "run", "--patient-twin", "inputs/patient_twin.yaml")

    assert _reported(output, "patient twin") == patient_twin
    metadata = json.loads((tmp_path / "run" / "run.json").read_text())
    assert metadata["patient_twin"] == str(patient_twin)


@pytest.mark.parametrize(
    ("endpoint", "transport_endpoint"),
    (
        ("192.0.2.10:7448", "tcp/192.0.2.10:7448"),
        ("192.0.2.10", "tcp/192.0.2.10:7448"),
        ("tcp/192.0.2.10:7448", "tcp/192.0.2.10:7448"),
    ),
)
def test_policy_endpoint_accepts_copy_pasteable_host_port(
    tmp_path: Path,
    endpoint: str,
    transport_endpoint: str,
) -> None:
    result = subprocess.run(
        [
            str(RUN_SH),
            WORKFLOW,
            "--policy",
            "--policy-endpoint",
            endpoint,
            "--dry-run",
            "--run-dir",
            str(tmp_path / "run"),
        ],
        cwd=WORKFLOWS_ROOT,
        capture_output=True,
        text=True,
        check=False,
    )

    assert result.returncode == 0, result.stderr
    assert f"==> external policy {transport_endpoint}" in result.stdout


def test_stop_requires_explicit_all_command() -> None:
    result = subprocess.run([str(STOP_SH), "status"], cwd=WORKFLOWS_ROOT, capture_output=True, text=True, check=False)

    assert result.returncode == 2
    assert result.stderr.strip() == "usage: ./stop.sh all"

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run.sh and stop.sh: workflow declarations, the commands they compose, and stopping."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("i4h_run_workflow", ROOT / "scripts" / "run_workflow.py")
run_workflow = importlib.util.module_from_spec(SPEC)
sys.modules[SPEC.name] = run_workflow
SPEC.loader.exec_module(run_workflow)

WORKFLOWS = run_workflow.load_workflows()


def _plan(*argv: str):
    return run_workflow.plan(list(argv))


def _served_tasks(stack: str) -> set[str]:
    tree = ast.parse((ROOT / "tasks" / stack / "i4h_tasks" / stack / "serve.py").read_text())
    for node in tree.body:
        if isinstance(node, ast.Assign) and any(getattr(t, "id", "") == "TASKS" for t in node.targets):
            return set(ast.literal_eval(node.value))
    raise AssertionError(f"{stack}/serve.py declares no TASKS")


def test_every_specialty_has_workflows() -> None:
    specialties = {workflow.specialty for workflow in WORKFLOWS.values()}
    assert specialties == {
        "laparoscopic-robotics",
        "ultrasound-robotics",
        "endoluminal-robotics",
        "hospital-automation-robotics",
    }


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_a_declaration_names_what_exists(name: str) -> None:
    workflow = WORKFLOWS[name]
    assert (run_workflow.SCENE_MANIFESTS / f"{workflow.scene}.yaml").is_file()
    scene_sources = "".join(path.read_text() for path in (ROOT / "arena" / "i4h_arena" / "scenes").glob("*.py"))
    for mode, spec in workflow.modes.items():
        policy = spec.get("policy", "")
        if policy.startswith("i4h_arena."):
            module, _, cls = policy.rpartition(".")
            source = ROOT / "arena" / Path(*module.split(".")).with_suffix(".py")
            assert f"class {cls}(" in source.read_text(), f"{name} {mode}: no {policy}"
        if "server" in spec:
            assert spec["server"]["task"] in _served_tasks(spec["server"]["stack"]), f"{name} {mode}"
        for device in spec.get("devices", []):
            assert f'"{device}"' in scene_sources, f"{name}: no scene declares teleop device {device!r}"


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_every_mode_composes_a_command(name: str) -> None:
    workflow = WORKFLOWS[name]
    for mode in workflow.mode_names():
        argv = [name]
        if mode == "replay":
            argv += ["--replay", "demos.hdf5"]
        elif mode == "teleop":
            argv += ["--teleop"]
        else:
            argv += ["--mode", mode]
        if workflow.modes.get(mode, {}).get("checkpoint") == "required":
            argv += ["--checkpoint", "/tmp/checkpoint"]
        built = _plan(*argv)
        assert f"i4h_arena.environments:{workflow.environment_class}" in built.command
        assert built.mode == mode


def test_a_served_policy_starts_its_stack_and_points_the_client_at_it() -> None:
    built = _plan("scissor_pick_and_place", "--policy", "--port", "6123", "--checkpoint", "/ckpt")
    assert built.server[1:] == [
        "-m",
        "i4h_tasks.gr00t_n15.serve",
        "scissor_pick_and_place",
        "--port",
        "6123",
        "--model-path",
        "/ckpt",
    ]
    command = built.command
    assert command[command.index("--remote_port") + 1] == "6123"
    # Policy and launcher options before the environment, the environment's own after it.
    assert command.index("--enable_cameras") < command.index("soarm_scissors")


def test_an_external_endpoint_starts_no_server() -> None:
    built = _plan("ultrasound_liver_scan", "--policy", "--policy-endpoint", "thor:8000")
    assert built.server is None
    command = built.command
    assert command[command.index("--remote_host") + 1] == "thor"
    assert command[command.index("--remote_port") + 1] == "8000"


def test_recording_paths_follow_the_caller(tmp_path, monkeypatch) -> None:
    monkeypatch.setenv("I4H_CALLER_CWD", str(tmp_path))
    built = _plan("surgical_reach_psm", "--rule-based", "--record", "out/verify.hdf5", "--headless")
    command = built.command
    assert command[command.index("--recorder_dataset_export_dir_path") + 1] == str(tmp_path / "out")
    assert command[command.index("--recorder_dataset_filename") + 1] == "verify"
    assert "--viz" not in command
    default = _plan("surgical_reach_psm", "--rule-based", "--record", "--run-dir", "r")
    assert default.record == tmp_path / "r" / "demos.hdf5"


def test_teleop_records_with_isaac_labs_tool() -> None:
    built = _plan(
        "scissor_pick_and_place", "--teleop", "so101_leader", "--episodes", "2", "--", "--leader_port", "/dev/x"
    )
    command = built.command
    assert command[1].endswith("scripts/tools/record_demos.py")
    assert command[command.index("--teleop_device") + 1] == "so101_leader"
    assert command[command.index("--num_demos") + 1] == "2"
    assert command[-2:] == ["--leader_port", "/dev/x"]


def test_the_tablecloth_is_teleoperated_through_xr_on_newton() -> None:
    built = _plan("spread_tablecloth_h2", "--teleop")
    assert "--teleop_device" not in built.command
    assert "--no-auto_launch_cloudxr" in built.command
    assert built.command[built.command.index("--presets") + 1] == "newton"
    assert built.env["PXR_WORK_THREAD_LIMIT"] == "1"


def test_launcher_options_go_before_the_environment_and_its_own_after() -> None:
    command = _plan("endoluminal_navigation", "--mode", "demo", "--headless").command
    assert command.index("--presets") < command.index("endoluminal_navigation") < command.index("--episode_steps")
    assert command[command.index("--episode_steps") + 1] == "192"


def test_without_a_mode_the_scene_opens_and_holds() -> None:
    built = _plan("surgical_reach_psm")
    assert built.mode == "idle"
    assert built.command[built.command.index("--policy_type") + 1] == run_workflow.HOLD_POLICY


def test_a_bare_scene_can_be_viewed_but_not_controlled() -> None:
    built = _plan("psm_reach", "--live")
    assert built.command[built.command.index("--policy_type") + 1] == run_workflow.HOLD_POLICY
    assert "--kit_args" in built.command
    with pytest.raises(run_workflow.UsageError, match="no mode 'rule-based'"):
        _plan("psm_reach", "--rule-based")


@pytest.mark.parametrize("option", ["--attempts", "--record-failures", "--prompt", "--python-server"])
def test_retired_options_say_what_replaces_them(option: str) -> None:
    with pytest.raises(run_workflow.UsageError, match="no longer supported"):
        _plan("surgical_reach_psm", option, "x")


def test_a_task_without_a_published_checkpoint_needs_one() -> None:
    with pytest.raises(run_workflow.UsageError, match="needs --checkpoint"):
        _plan("endoluminal_navigation", "--mode", "policy_n17")


def test_interactive_modes_need_a_window() -> None:
    with pytest.raises(run_workflow.UsageError, match="visible Kit session"):
        _plan("scissor_pick_and_place", "--teleop", "--headless")


def test_stop_ends_only_the_runs_this_checkout_recorded(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_workflow, "PID_DIR", tmp_path)
    ours = subprocess.Popen(["sleep", "60"], start_new_session=True)
    theirs = subprocess.Popen(["sleep", "60"], start_new_session=True)
    try:
        (tmp_path / f"surgical_reach_psm.{os.getpid() + 10_000_000}").write_text(f"{ours.pid}\n")
        assert run_workflow.stop("all") == 0
        deadline = time.monotonic() + 10
        while ours.poll() is None and time.monotonic() < deadline:
            time.sleep(0.1)
        assert ours.poll() is not None
        assert theirs.poll() is None
        assert not list(tmp_path.iterdir())
    finally:
        theirs.kill()
        ours.kill()


def test_a_stopped_child_is_reaped_rather_than_waited_out() -> None:
    """An exited but unreaped leader keeps its group alive until the last grace period ends."""
    child = subprocess.Popen(["sleep", "60"], start_new_session=True)
    started = time.monotonic()
    run_workflow._stop_group(child.pid, grace_s=5.0, process=child)
    assert time.monotonic() - started < 3.0
    assert child.returncode is not None


def _orphan() -> int:
    """A ``sleep`` in its own session whose parent has exited, as run.sh's processes are to stop.sh."""
    spawn = (
        "import subprocess; "
        "print(subprocess.Popen(['sleep', '60'], start_new_session=True,"
        " stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL).pid)"
    )
    return int(subprocess.run([sys.executable, "-c", spawn], capture_output=True, text=True, check=True).stdout)


def test_stop_asks_the_launcher_first(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(run_workflow, "PID_DIR", tmp_path)
    launcher, group = _orphan(), _orphan()
    try:
        (tmp_path / f"surgical_reach_psm.{launcher}").write_text(f"{group}\n")
        started = time.monotonic()
        assert run_workflow.stop("surgical_reach_psm") == 0
        assert time.monotonic() - started < 10.0
        assert not run_workflow._running(launcher)
        assert not run_workflow._running(group)
    finally:
        for pid in (launcher, group):
            with contextlib.suppress(ProcessLookupError):
                os.killpg(pid, signal.SIGKILL)

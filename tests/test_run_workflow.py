# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""run.sh and stop.sh: workflow declarations, the commands they compose, and stopping."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import json
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
    # run.sh passes the declared presets, so launcher arguments carrying them would pass two.
    assert "--presets" not in workflow.launcher_args, f"{name}: declare presets: [...] instead"
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
def test_every_mode_composes_a_command(name: str, tmp_path) -> None:
    workflow = WORKFLOWS[name]
    for mode in workflow.mode_names():
        argv = [name]
        if mode == "replay":
            import h5py

            with h5py.File(tmp_path / "demos.hdf5", "w") as handle:
                handle.create_group("data")
            if name in ("endoluminal_navigation", "endoluminal_navigation_arm"):
                argv += ["--catheter-backend", "legacy"]
            argv += ["--replay", str(tmp_path / "demos.hdf5")]
        elif mode == "teleop":
            argv += ["--teleop"]
        else:
            argv += ["--mode", mode]
        spec = workflow.modes.get(mode, {})
        if spec.get("policy") == "rsl_rl":
            argv += ["--checkpoint", str(_rsl_rl_training_run(tmp_path / mode))]
        elif spec.get("checkpoint") == "required":
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


def test_a_bare_scene_runs_with_the_presets_and_environment_of_its_workflow() -> None:
    built = _plan("g1_tablecloth", "--headless")
    assert built.command[built.command.index("--presets") + 1] == "newton"
    assert built.env["PXR_WORK_THREAD_LIMIT"] == "1"
    with pytest.raises(run_workflow.UsageError, match="runs only with --presets newton, not physx"):
        _plan("h2_tablecloth", "--presets", "physx")
    assert "--presets" not in _plan("psm_reach").command


def test_a_bare_scene_takes_only_what_all_its_workflows_agree_on(monkeypatch) -> None:
    declared = {
        "a": run_workflow.Workflow(name="a", scene="psm_reach", presets=["newton"], env={"X": "1", "Y": "1"}),
        "b": run_workflow.Workflow(name="b", scene="psm_reach", presets=["physx"], env={"X": "1", "Y": "2"}),
    }
    monkeypatch.setattr(run_workflow, "load_workflows", lambda: declared)

    bare = run_workflow.resolve("psm_reach")

    assert bare.presets == []
    assert bare.env == {"X": "1"}


@pytest.mark.parametrize("option", ["--attempts", "--record-failures", "--prompt", "--python-server"])
def test_retired_options_say_what_replaces_them(option: str) -> None:
    with pytest.raises(run_workflow.UsageError, match="no longer supported"):
        _plan("surgical_reach_psm", option, "x")


@pytest.mark.parametrize("argv", [["lint"], ["lint", "--all"], ["lint", "surgical_reach_psm"]])
def test_retired_subcommands_say_what_replaces_them(argv: list[str], capsys) -> None:
    assert run_workflow.main(argv) == 2
    error = capsys.readouterr().err
    assert error.startswith("run.sh: lint is no longer supported")
    assert "./run.sh show" in error
    assert "tests/test_run_workflow.py" in error


def test_a_scene_bound_to_one_physics_backend_rejects_another_before_launch() -> None:
    with pytest.raises(run_workflow.UsageError, match="runs only with --presets newton, not physx"):
        _plan("spread_tablecloth_g1", "--presets", "physx")
    assert _plan("spread_tablecloth_g1", "--presets", "newton").command.count("--presets") == 1


@pytest.mark.parametrize("name", sorted(WORKFLOWS))
def test_presets_are_passed_once(name: str) -> None:
    preset = (WORKFLOWS[name].presets or ["newton"])[0]
    command = _plan(name, "--presets", preset).command
    assert command.count("--presets") == 1
    assert command[command.index("--presets") + 1] == preset


def _rsl_rl_training_run(run: Path) -> Path:
    """The layout ./train.sh rl leaves: Isaac Lab's log directory, run.json, model_final.pt, exported/."""
    log = run / "2026-10-22_21-14-13"
    (log / "params").mkdir(parents=True)
    (log / "params" / "agent.yaml").write_text("seed: 42\n")
    checkpoints = [log / f"model_{iteration}.pt" for iteration in (0, 25, 399)]
    for checkpoint in checkpoints:
        checkpoint.write_bytes(b"")
    (run / "model_final.pt").write_bytes(b"")
    (run / "run.json").write_text(json.dumps({"checkpoints": [str(path) for path in checkpoints]}))
    (run / "exported").mkdir()
    (run / "exported" / "policy.pt").write_bytes(b"")
    (run / "exported" / "policy.json").write_text(json.dumps({"source_checkpoint": str(run / "model_final.pt")}))
    return checkpoints[-1]


@pytest.mark.parametrize(
    "given", ["", "model_final.pt", "exported", "exported/policy.pt", "2026-10-22_21-14-13/model_399.pt"]
)
def test_an_rsl_rl_checkpoint_runs_as_its_training_checkpoint(tmp_path, given: str) -> None:
    run = tmp_path / "run"
    last = _rsl_rl_training_run(run)
    built = _plan("ultrasound_probe_reach", "--policy", "--checkpoint", str(run / given))
    command = built.command
    assert command[command.index("--checkpoint_path") + 1] == str(last)
    assert bool(built.notes) == (run / given != last)


def test_an_rsl_rl_checkpoint_without_its_agent_config_is_rejected_before_launch(tmp_path) -> None:
    lone = tmp_path / "policy.pt"
    lone.write_bytes(b"")
    with pytest.raises(run_workflow.UsageError, match="params/agent.yaml beside it"):
        _plan("ultrasound_probe_reach", "--policy", "--checkpoint", str(lone))
    with pytest.raises(run_workflow.UsageError, match="does not exist"):
        _plan("ultrasound_probe_reach", "--policy", "--checkpoint", str(tmp_path / "missing.pt"))


def _episode_results(built, *successes) -> None:
    outputs = built.output_dir / "2026-10-23_00-00-00"
    outputs.mkdir(parents=True)
    lines = "".join(json.dumps({"success": success}) + "\n" for success in successes)
    (outputs / "episode_results_rank0.jsonl").write_text(lines)


def test_the_summary_counts_the_episodes_the_success_term_judged(tmp_path) -> None:
    built = _plan("surgical_reach_psm", "--rule-based", "--episodes", "2", "--run-dir", str(tmp_path))
    _episode_results(built, True, False)
    assert run_workflow._summary(built) == "1/2 episodes succeeded"


def test_a_mode_the_success_term_does_not_judge_reports_the_episodes_it_ran(tmp_path) -> None:
    argv = ["--mode", "validate_fluoroscopy", "--episodes", "2", "--run-dir", str(tmp_path)]
    built = _plan("endoluminal_navigation", *argv)
    _episode_results(built, False, False)
    assert run_workflow._summary(built) == (
        "2 episode(s) ran; validate_fluoroscopy episodes are not scored by the scene's success term"
    )


def test_the_summary_says_when_a_run_ended_before_its_episodes(tmp_path) -> None:
    built = _plan("surgical_reach_psm", "--rule-based", "--episodes", "3", "--run-dir", str(tmp_path))
    _episode_results(built, True)
    assert run_workflow._summary(built) == "1/1 episodes succeeded (1 of 3 requested)"


def test_every_third_party_patch_is_applied_by_setup() -> None:
    setup = (ROOT / "third_party" / "setup.sh").read_text() + (ROOT / "scripts" / "setup_rl_cpu.sh").read_text()
    for patch in sorted((ROOT / "third_party").glob("*.patch")):
        assert patch.name in setup, f"{patch.name} is not applied by third_party/setup.sh"


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


def test_replay_names_a_missing_recording_before_anything_starts(tmp_path) -> None:
    with pytest.raises(run_workflow.UsageError, match="no recording at"):
        _plan("surgical_reach_psm", "--replay", str(tmp_path / "absent.hdf5"))


def test_declared_tool_path_is_independent_of_checkout_globbing(tmp_path, monkeypatch):
    from i4h_common.paths import package_source

    declared = package_source("isaaclab-dev", ROOT) / "scripts/tools"
    assert str(declared / "record_demos.py") in _plan("scissor_pick_and_place", "--teleop").command
    pinned = tmp_path / "pinned/scripts/tools"
    monkeypatch.setattr(run_workflow, "package_source", lambda *args: pinned.parent.parent)
    assert _plan("scissor_pick_and_place", "--teleop").command[1] == str(pinned / "record_demos.py")
    recording = tmp_path / "input.hdf5"
    recording.touch()
    assert _plan("scissor_pick_and_place", "--replay", str(recording)).command[1] == str(pinned / "replay_demos.py")


def test_execution_reports_missing_setup_before_reserving_directory(tmp_path, monkeypatch):
    built = _plan("scissor_pick_and_place", "--teleop", "--run-dir", str(tmp_path / "run"))
    built.command[0] = sys.executable
    built.command[1] = str(tmp_path / "missing/record_demos.py")
    with pytest.raises(run_workflow.UsageError, match="missing pinned Isaac Lab tool.*setup.sh"):
        run_workflow.run(built)
    assert not built.run_dir.exists()
    signal.signal(signal.SIGINT, signal.default_int_handler)
    signal.signal(signal.SIGTERM, signal.SIG_DFL)


def test_same_timestamp_runs_reserve_unique_directories_and_keep_independent_results(tmp_path):
    from concurrent.futures import ThreadPoolExecutor
    from dataclasses import replace

    first = _plan("surgical_reach_psm", "--rule-based", "--record", "--run-dir", str(tmp_path / "20261120_120000"))
    plans = [replace(first, command=first.command.copy(), default_run_dir=True) for _ in range(8)]
    for built in plans:
        built.default_record = True
    # Synchronize reservations, as eight launchers planning within the same second would do.
    from threading import Barrier

    barrier = Barrier(len(plans))

    def reserve(built):
        barrier.wait()
        run_workflow._reserve_run_directory(built)
        child = (
            "import pathlib,sys,json; p=pathlib.Path(sys.argv[1]); p.mkdir(parents=True);"
            " (p/'episode_results_rank0.jsonl').write_text(json.dumps({'success': True}) + chr(10))"
        )
        subprocess.run([sys.executable, "-c", child, str(built.output_dir / "fake")], check=True)
        run_workflow._write_recording_metadata(built)

    with ThreadPoolExecutor(max_workers=8) as pool:
        list(pool.map(reserve, plans))
    assert len({built.run_dir for built in plans}) == 8
    for built in plans:
        assert built.record == built.run_dir / "demos.hdf5"
        assert str(built.run_dir / "outputs") in built.command
        assert str(built.run_dir) in built.command
        assert run_workflow._summary(built) == "1/1 episodes succeeded"
        assert json.loads((built.record.parent / "run.json").read_text())["recording"] == str(built.record)


def test_previews_do_not_reserve_and_explicit_paths_survive_reservation(tmp_path):
    built = _plan(
        "surgical_reach_psm",
        "--rule-based",
        "--record",
        str(tmp_path / "external/data.hdf5"),
        "--run-dir",
        str(tmp_path / "explicit"),
    )
    run_workflow.show(built)
    assert not built.run_dir.exists() and not built.record.parent.exists()
    run_workflow._reserve_run_directory(built)
    assert built.run_dir == tmp_path / "explicit"
    assert built.record == tmp_path / "external/data.hdf5"
    built.default_run_dir = True
    run_workflow._reserve_run_directory(built)
    assert built.run_dir == tmp_path / "explicit_1"
    assert built.record == tmp_path / "external/data.hdf5"


def test_external_recording_metadata_rejects_conflicting_patient_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(run_workflow, "installed_numerical_revision", lambda: "candidate-revision")
    twin = tmp_path / "patient.yaml"
    twin.write_text("patient_id: first\n")
    built = _plan(
        "endoluminal_navigation",
        "--mode",
        "demo",
        "--patient-twin",
        str(twin),
        "--record",
        str(tmp_path / "external/data.hdf5"),
        "--run-dir",
        str(tmp_path / "run"),
    )
    run_workflow._write_recording_metadata(built)
    sidecar = built.record.parent / "run.json"
    metadata = json.loads(sidecar.read_text())
    assert metadata["patient_twin"] == str(twin.resolve())
    assert metadata["recording"] == str(built.record)
    run_workflow._write_recording_metadata(built)
    twin.write_text("patient_id: changed\n")
    with pytest.raises(run_workflow.UsageError, match="conflicting.*provenance"):
        run_workflow._write_recording_metadata(built)
    assert json.loads(sidecar.read_text()) == metadata


def test_missing_patient_manifest_does_not_publish_provenance(tmp_path):
    built = _plan(
        "endoluminal_navigation",
        "--mode",
        "demo",
        "--patient-twin",
        str(tmp_path / "missing.yaml"),
        "--record",
        str(tmp_path / "out/recording.hdf5"),
    )
    with pytest.raises(run_workflow.UsageError, match="manifest is missing"):
        run_workflow._write_recording_metadata(built)
    assert not (built.record.parent / "run.json").exists()


def test_explicit_recording_under_original_outputs_keeps_its_path_on_collision(tmp_path):
    original = tmp_path / "20261120_120000"
    explicit = original / "outputs/custom.hdf5"
    built = _plan("surgical_reach_psm", "--rule-based", "--run-dir", str(original), "--record", str(explicit))
    built.default_run_dir = True
    original.mkdir()
    run_workflow._reserve_run_directory(built)
    assert built.run_dir == original.with_name(original.name + "_1")
    assert built.record == explicit
    assert built.command[built.command.index("--output_base_dir") + 1] == str(built.run_dir / "outputs")
    assert built.command[built.command.index("--recorder_dataset_export_dir_path") + 1] == str(explicit.parent)


def test_missing_lab_source_is_reported_only_for_modes_that_need_tools(monkeypatch):
    def missing(*args):
        raise KeyError("isaaclab-dev")

    monkeypatch.setattr(run_workflow, "package_source", missing)
    assert _plan("surgical_reach_psm", "--rule-based").mode == "rule-based"
    with pytest.raises(run_workflow.UsageError, match="declare a valid path source for isaaclab-dev"):
        _plan("scissor_pick_and_place", "--teleop")


@pytest.mark.parametrize("teleop", [True, False])
def test_default_record_tracks_automatic_and_explicit_paths(teleop, tmp_path):
    mode = ["--teleop"] if teleop else ["--rule-based", "--record"]
    workflow = "scissor_pick_and_place" if teleop else "surgical_reach_psm"
    assert _plan(workflow, *mode).default_record
    explicit = ["--teleop", "--record"] if teleop else mode
    assert not _plan(workflow, *explicit, str(tmp_path / "explicit.hdf5")).default_record


@pytest.mark.parametrize("workflow", ["endoluminal_navigation", "endoluminal_navigation_arm"])
def test_backend_is_forwarded_through_every_navigation_mode(workflow, tmp_path):
    import h5py

    source = tmp_path / "legacy.hdf5"
    with h5py.File(source, "w") as handle:
        handle.create_group("data").attrs["env_args"] = json.dumps({"catheter_backend": "legacy"})
    for mode in WORKFLOWS[workflow].mode_names():
        args = [workflow, "--catheter-backend", "legacy"]
        args += ["--replay", str(source)] if mode == "replay" else ["--mode", mode]
        if WORKFLOWS[workflow].modes.get(mode, {}).get("checkpoint"):
            args += ["--checkpoint", "/tmp/checkpoint"]
        built = _plan(*args)
        assert built.catheter_backend == "legacy"
        offset = built.command.index("--catheter_backend")
        assert built.command[offset + 1] == "legacy"

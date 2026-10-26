#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run and stop i4h workflows: the implementation behind ``run.sh`` and ``stop.sh``.

A workflow is declared by ``workflows/<specialty>/<workflow>.yaml``: the scene it runs
and, per mode, the controller, policy server, or teleoperation device. Each mode maps
to the tool an Isaac Lab or IsaacLab-Arena user would run:

- rule-based and other controller modes, ``idle`` and ``live``: Arena's policy runner;
- ``policy`` modes: the policy stack's ``serve`` module, then Arena's policy runner
  with the client policy;
- ``teleop``: Isaac Lab's ``record_demos.py``;
- ``replay``: Isaac Lab's ``replay_demos.py``.

``run.sh show`` prints those commands without running them. Every process a run
starts is recorded under ``.run/pids.d/``, which ``stop.sh`` reads, so a stop never
reaches another session's simulator or server.
"""

from __future__ import annotations

import argparse
import importlib.util
import json
import os
import platform
import shlex
import signal
import socket
import subprocess
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[1]
WORKFLOWS = ROOT / "workflows"
SCENE_MANIFESTS = ROOT / "arena" / "i4h_arena" / "scenes" / "manifest"
ARENA_PYTHON = ROOT / "arena" / ".venv" / "bin" / "python"
ISAACLAB = next(iter(sorted(ROOT.glob("third_party/IsaacLab-*/scripts/tools"))), ROOT / "third_party")
REGISTRATION_CALLBACK = "isaaclab_arena.environments.isaaclab_interop.environment_registration_callback"
PID_DIR = Path(os.environ.get("I4H_PID_DIR", ROOT / ".run" / "pids.d"))

#: Each stack's default port and the option its ``serve`` module takes a checkpoint by.
STACKS = {
    "gr00t_n15": (5555, "--model-path"),
    "gr00t_n16": (5555, "--model-path"),
    "gr00t_n17": (5555, "--model-path"),
    "openpi_pi0": (8000, "--checkpoint"),
}
#: Modes every workflow has; ``teleop`` comes from its declaration.
BUILTIN_MODES = ("idle", "live", "replay")
HOLD_POLICY = "i4h_arena.policies.hold.HoldPolicy"
#: Kit extensions a live session needs for scripts/live_scene_edit.py.
LIVE_KIT_ARGS = (
    "--enable isaacsim.code_editor.python_server --enable isaacsim.test.utils "
    "--enable isaacsim.core.rendering_manager"
)
#: Long enough that a live or idle session never resets under an edit.
UNBOUNDED_STEPS = "1000000000"

#: Options of the earlier launcher whose behaviour Arena does not have, and what to do instead.
RETIRED = {
    "--attempts": "every episode runs once; ask for more with --episodes",
    "--max-attempts": "every episode runs once; ask for more with --episodes",
    "--record-failures": (
        "a policy run records every episode with its success flag; filter with i4h-dataset convert --successful-only"
    ),
    "--save-all-episodes": "a policy run records every episode with its success flag",
    "--prompt": "the policy is prompted with the environment's task description",
    "--namespace": "runs no longer share a bus; give a second run its own --port",
    "--view-sensor": "a scene opens its sensor panels itself in a visible run",
    "--no-sensor-view": "a scene opens its sensor panels itself in a visible run",
    "--python-server": "use --live",
    "--fluoro-device": "the endoluminal environments render fluoroscopy on Vulkan",
}
#: Subcommands of the earlier launcher, and what to do instead.
RETIRED_COMMANDS = {
    "lint": (
        "a workflow is a YAML declaration now; `./run.sh show <workflow> [--<mode>]` prints the commands a mode "
        "runs, and tests/test_run_workflow.py checks that every declaration names what exists"
    ),
}


class UsageError(Exception):
    pass


# -- declarations ---------------------------------------------------------------


@dataclass
class Workflow:
    name: str
    scene: str
    specialty: str = ""
    description: str = ""
    launcher_args: list[str] = field(default_factory=list)
    environment_args: list[str] = field(default_factory=list)
    env: dict[str, str] = field(default_factory=dict)
    #: The physics presets the scene runs with, the first by default; empty for any.
    presets: list[str] = field(default_factory=list)
    modes: dict[str, dict[str, Any]] = field(default_factory=dict)

    @property
    def environment_class(self) -> str:
        return "".join(part.capitalize() for part in self.scene.split("_"))

    def mode_names(self) -> list[str]:
        return [*self.modes, *(mode for mode in BUILTIN_MODES if mode not in self.modes)]


def _yaml(path: Path) -> Any:
    import yaml  # noqa: PLC0415 - stop.sh runs without the common environment

    return yaml.safe_load(path.read_text())


def _scene_manifest(scene: str) -> dict[str, Any] | None:
    path = SCENE_MANIFESTS / f"{scene}.yaml"
    return _yaml(path) if path.is_file() else None


def load_workflows(root: Path = WORKFLOWS) -> dict[str, Workflow]:
    workflows: dict[str, Workflow] = {}
    for path in sorted(root.glob("*/*.yaml")):
        raw = _yaml(path) or {}
        workflows[path.stem] = Workflow(
            name=path.stem,
            scene=str(raw["scene"]),
            specialty=path.parent.name,
            description=str(raw.get("description", "")),
            launcher_args=[str(arg) for arg in raw.get("launcher_args", [])],
            environment_args=[str(arg) for arg in raw.get("environment_args", [])],
            env={str(key): str(value) for key, value in (raw.get("env") or {}).items()},
            presets=[str(preset) for preset in raw.get("presets", [])],
            modes={str(name): dict(spec or {}) for name, spec in (raw.get("modes") or {}).items()},
        )
    return workflows


def resolve(name: str) -> Workflow:
    """A declared workflow, or a bare scene, which can only be viewed, edited live, or replayed.

    A bare scene runs with the presets and process environment that every workflow on it
    declares, such as Newton and ``PXR_WORK_THREAD_LIMIT=1`` for the tablecloth scenes.
    """
    workflows = load_workflows()
    if name in workflows:
        return workflows[name]
    if _scene_manifest(name) is not None:
        declared = [workflow for workflow in workflows.values() if workflow.scene == name]
        presets = declared[0].presets if declared and all(w.presets == declared[0].presets for w in declared) else []
        env = dict(set.intersection(*(set(w.env.items()) for w in declared))) if declared else {}
        return Workflow(
            name=name, scene=name, description="scene without a workflow declaration", presets=presets, env=env
        )
    known = ", ".join(sorted(workflows))
    raise UsageError(f"unknown workflow or scene {name!r}; workflows: {known}")


# -- command line ---------------------------------------------------------------


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="run.sh", add_help=False)
    parser.add_argument("workflow")
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--mode")
    modes.add_argument("--policy", dest="mode", action="store_const", const="policy")
    modes.add_argument("--rule-based", dest="mode", action="store_const", const="rule-based")
    modes.add_argument("--idle", dest="mode", action="store_const", const="idle")
    modes.add_argument("--live", dest="mode", action="store_const", const="live")
    modes.add_argument("--teleop", nargs="?", const="", metavar="DEVICE")
    modes.add_argument("--replay", metavar="PATH")
    parser.add_argument("--episodes", type=int)
    parser.add_argument("--episode", "--episode-index", dest="episode", type=int)
    parser.add_argument("--episode-steps", "--timesteps", dest="episode_steps", type=int)
    parser.add_argument("--record", "--record-to", dest="record", nargs="?", const="", metavar="PATH")
    parser.add_argument("--run-dir")
    parser.add_argument("--checkpoint")
    parser.add_argument("--policy-endpoint", metavar="HOST:PORT")
    parser.add_argument("--port", type=int)
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--presets")
    parser.add_argument("--no-cameras", action="store_true")
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--patient-twin")
    parser.add_argument("--fluoro-backend")
    parser.add_argument("--cloudxr-env")
    parser.add_argument("--dry-run", action="store_true")
    return parser


@dataclass
class Plan:
    workflow: Workflow
    mode: str
    run_dir: Path
    command: list[str]
    env: dict[str, str]
    server: list[str] | None = None
    port: int | None = None
    record: Path | None = None
    output_dir: Path | None = None
    episodes: int | None = None
    #: False for a mode whose episodes the scene's success term does not judge.
    judged: bool = True
    notes: list[str] = field(default_factory=list)


def _caller_path(path: str) -> Path:
    caller = Path(os.environ.get("I4H_CALLER_CWD", os.getcwd()))
    return (caller / path).resolve() if not os.path.isabs(path) else Path(path)


def _free_port(preferred: int) -> int:
    for port in (preferred, *range(preferred + 1, preferred + 50)):
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) != 0:
                return port
    raise RuntimeError(f"no free port near {preferred}")


def _json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text())


def _rsl_rl_checkpoint(path: Path) -> Path:
    """The ``model_<N>.pt`` that ``path`` stands for. Arena's ``rsl_rl`` policy loads a
    checkpoint with the ``params/agent.yaml`` Isaac Lab's ``train.py`` wrote beside it.
    ``./train.sh rl`` also leaves a run directory whose ``run.json`` lists those
    checkpoints, a ``model_final.pt`` copy of the last one, and ``exported/``, whose
    ``policy.json`` names the checkpoint it was exported from."""
    named = path
    for _ in range(4):
        if path.is_file() and (path.parent / "params" / "agent.yaml").is_file():
            return path
        folder = path if path.is_dir() else path.parent
        if (folder / "policy.json").is_file() and (path.is_dir() or path.name == "policy.pt"):
            source = _json(folder / "policy.json").get("source_checkpoint")
        elif (folder / "run.json").is_file() and (path.is_dir() or path.name == "model_final.pt"):
            source = (_json(folder / "run.json").get("checkpoints") or [None])[-1]
        else:
            source = None
        if not source:
            break
        path = Path(source)
    if not path.exists():
        raise UsageError(f"--checkpoint {named}: {path} does not exist")
    raise UsageError(
        f"--checkpoint {named}: Arena's rsl_rl policy needs a model_<N>.pt with params/agent.yaml beside it; "
        "pass the training run directory, its model_final.pt or exported/policy.pt, or a model_<N>.pt"
    )


def plan(argv: list[str]) -> Plan:
    if "--" in argv:
        split = argv.index("--")
        argv, extra = argv[:split], argv[split + 1 :]
    else:
        extra = []
    for option in argv:
        name = option.split("=", 1)[0]
        if name in RETIRED:
            raise UsageError(f"{name} is no longer supported: {RETIRED[name]}")
    args = _parser().parse_args(argv)
    workflow = resolve(args.workflow)

    if args.teleop is not None:
        mode = "teleop"
    elif args.replay is not None:
        mode = "replay"
    else:
        mode = args.mode or "idle"
    if mode not in workflow.mode_names():
        raise UsageError(f"{workflow.name} has no mode {mode!r}; it has {', '.join(workflow.mode_names())}")
    spec = workflow.modes.get(mode, {})

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = _caller_path(args.run_dir) if args.run_dir else ROOT / "runs" / workflow.name / stamp
    record = None
    if args.record is not None:
        record = _caller_path(args.record) if args.record else run_dir / "demos.hdf5"
        if record.suffix != ".hdf5":
            raise UsageError(f"--record names an .hdf5 file, not {record}")

    env = {"OMNI_KIT_ACCEPT_EULA": os.environ.get("OMNI_KIT_ACCEPT_EULA", "YES"), **workflow.env}
    visible = not args.headless
    if mode in ("teleop", "live") and not visible:
        raise UsageError(f"{mode} needs a visible Kit session; drop --headless")
    cameras = (
        not args.no_cameras
        and spec.get("cameras", True)
        and bool((_scene_manifest(workflow.scene) or {}).get("cameras"))
    )

    presets = args.presets
    if workflow.presets:
        presets = presets or workflow.presets[0]
        if presets not in workflow.presets:
            raise UsageError(f"{workflow.name} runs only with --presets {' or '.join(workflow.presets)}, not {presets}")
    launcher = [*workflow.launcher_args]
    for option, value in (("--presets", presets), ("--seed", args.seed), ("--device", args.device)):
        if value is not None:
            launcher += [option, str(value)]
    environment = [*workflow.environment_args, *[str(arg) for arg in spec.get("environment_args", [])]]
    if args.episode_steps is not None:
        environment += ["--episode_steps", str(args.episode_steps)]
    if args.patient_twin:
        environment += ["--patient_twin", str(_caller_path(args.patient_twin))]
    if args.fluoro_backend:
        environment += ["--fluoro_backend", args.fluoro_backend]

    if mode in ("teleop", "replay"):
        command, record = _isaaclab_tool(
            workflow, mode, spec, args, launcher, environment, record, run_dir, visible, cameras
        )
        return Plan(workflow, mode, run_dir, [*command, *extra], env, record=record)

    server = None
    port = None
    policy_args = [str(arg) for arg in spec.get("args", [])]
    if mode in ("idle", "live"):
        policy = HOLD_POLICY
        environment += ["--episode_steps", UNBOUNDED_STEPS]
        if mode == "live":
            launcher += ["--kit_args", LIVE_KIT_ARGS]
    else:
        policy = str(spec["policy"])
    checkpoint = _caller_path(args.checkpoint) if args.checkpoint else None
    if spec.get("checkpoint") == "required" and checkpoint is None and not args.policy_endpoint:
        raise UsageError(f"{workflow.name} {mode} needs --checkpoint: its task names no published checkpoint")
    notes: list[str] = []
    if checkpoint is not None and policy == "rsl_rl":
        native = _rsl_rl_checkpoint(checkpoint)
        if native != checkpoint:
            notes.append(f"--checkpoint {checkpoint} runs as its training checkpoint {native}")
        checkpoint = native
    if "server" in spec:
        stack, task = str(spec["server"]["stack"]), str(spec["server"]["task"])
        default_port, checkpoint_option = STACKS[stack]
        if args.policy_endpoint:
            host, _, endpoint_port = args.policy_endpoint.rpartition(":")
            policy_args += ["--remote_host", host or "localhost", "--remote_port", endpoint_port]
        else:
            port = args.port or _free_port(default_port)
            server = [str(ROOT / "tasks" / stack / ".venv" / "bin" / "python"), "-m", f"i4h_tasks.{stack}.serve", task]
            server += ["--port", str(port)]
            if checkpoint is not None:
                server += [checkpoint_option, str(checkpoint)]
            policy_args += ["--remote_port", str(port)]
    elif checkpoint is not None:
        option = spec.get("checkpoint_arg")
        if not option:
            raise UsageError(f"{workflow.name} {mode} takes no --checkpoint")
        policy_args += [str(option), str(checkpoint)]

    output_dir = run_dir / "outputs"
    command = [str(ARENA_PYTHON), "-m", "isaaclab_arena.evaluation.policy_runner", "--policy_type", policy]
    command += policy_args
    if mode in ("idle", "live"):
        command += ["--num_steps", UNBOUNDED_STEPS]
    else:
        command += ["--num_episodes", str(args.episodes or 1)]
    command += launcher
    if cameras:
        command += ["--enable_cameras"]
    if visible:
        command += ["--viz", "kit"]
    command += ["--output_base_dir", str(output_dir)]
    command += ["--external_environment_class_path", f"i4h_arena.environments:{workflow.environment_class}"]
    command += [workflow.scene, *environment]
    if record is not None:
        command += [
            "--recorder_dataset_export_dir_path",
            str(record.parent),
            "--recorder_dataset_filename",
            record.stem,
        ]
    command += extra
    return Plan(
        workflow,
        mode,
        run_dir,
        command,
        env,
        server,
        port,
        record,
        output_dir,
        args.episodes or 1,
        judged=spec.get("success_term", True),
        notes=notes,
    )


def _isaaclab_tool(workflow, mode, spec, args, launcher, environment, record, run_dir, visible, cameras):
    command = [str(ARENA_PYTHON), str(ISAACLAB / ("record_demos.py" if mode == "teleop" else "replay_demos.py"))]
    command += ["--external_callback", REGISTRATION_CALLBACK]
    command += ["--external_environment_class_path", f"i4h_arena.environments:{workflow.environment_class}"]
    command += ["--task", workflow.scene, *launcher]
    if visible:
        command += ["--viz", "kit"]
    if mode == "teleop":
        if spec.get("xr"):
            if args.teleop:
                raise UsageError(f"{workflow.name} is teleoperated through Isaac Teleop and takes no device")
            command += (
                ["--cloudxr_env", str(_caller_path(args.cloudxr_env))]
                if args.cloudxr_env
                else ["--no-auto_launch_cloudxr"]
            )
        else:
            devices = [str(device) for device in spec.get("devices", [])]
            device = args.teleop or devices[0]
            if device not in devices:
                raise UsageError(f"{workflow.name} has no teleop device {device!r}; it has {', '.join(devices)}")
            command += ["--teleop_device", device]
        if cameras:
            command += ["--enable_cameras"]
        record = record or run_dir / "teleop.hdf5"
        command += ["--dataset_file", str(record)]
        if args.episodes:
            command += ["--num_demos", str(args.episodes)]
    else:
        record = None
        command += ["--dataset_file", str(_caller_path(args.replay))]
        if args.episode is not None:
            command += ["--select_episodes", str(args.episode)]
    command += [*[str(arg) for arg in spec.get("args", [])], *environment]
    return command, record


# -- running --------------------------------------------------------------------


def _quote(command: list[str]) -> str:
    return " ".join(shlex.quote(part) for part in command)


def show(plan: Plan) -> None:
    env = " ".join(f"{key}={shlex.quote(value)}" for key, value in plan.workflow.env.items())
    for note in plan.notes:
        print(f"# {note}")
    if plan.server:
        print(f"# policy server ({plan.workflow.name} {plan.mode}), started first and stopped after:")
        print(_quote(plan.server))
    print(f"# {plan.workflow.name} [{plan.mode}] as {plan.workflow.environment_class} ({plan.workflow.scene}):")
    print(f"{env + ' ' if env else ''}{_quote(plan.command)}")


def _arena_env(extra: dict[str, str]) -> dict[str, str]:
    env = {key: value for key, value in os.environ.items() if key != "VIRTUAL_ENV"}
    env.update(extra)
    if platform.machine() == "aarch64":
        spec = importlib.util.spec_from_file_location("i4h_rl_runtime", ROOT / "rl" / "i4h_rl" / "runtime.py")
        runtime = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(runtime)
        runtime.apply_arm_openmp_env(env)
    return env


def _decline_telemetry() -> None:
    """Answer Kit's telemetry prompt, which the EULA variable does not, unless the user has."""
    privacy = Path.home() / ".nvidia-omniverse" / "config" / "privacy.toml"
    if not privacy.exists():
        privacy.parent.mkdir(parents=True, exist_ok=True)
        privacy.write_text("[privacy]\nperformance = false\npersonalization = false\nusage = false\n")
        print(f"==> wrote {privacy} (declining telemetry; edit to opt in)", flush=True)


def _wait_for_port(port: int, server: subprocess.Popen, log: Path) -> None:
    deadline = time.monotonic() + float(os.environ.get("I4H_SERVER_TIMEOUT_S", "3600"))
    while True:
        with socket.socket() as sock:
            if sock.connect_ex(("127.0.0.1", port)) == 0:
                return
        if server.poll() is not None:
            raise RuntimeError(f"the policy server exited ({server.returncode}) before it listened; see {log}")
        if time.monotonic() > deadline:
            raise RuntimeError(f"the policy server did not listen on {port}; see {log}")
        time.sleep(2)


#: SIGINT, as Ctrl+C sends it, a few times before SIGTERM. Kit exits within seconds of a
#: KeyboardInterrupt in the runner's loop, but one raised inside a Kit event callback is
#: reported and dropped; SIGTERM can take longer than the grace period to act on.
STOP_SIGNALS = (*[(signal.SIGINT, 2.0)] * 5, (signal.SIGTERM, None), (signal.SIGKILL, None))


def _stop_group(pid: int, grace_s: float = 20.0, process: subprocess.Popen | None = None) -> None:
    """Stop a process group started with its own session, by ``STOP_SIGNALS``. Pass the
    leader's ``Popen`` when this process started it, so that it is reaped afterwards."""
    for sig, wait_s in STOP_SIGNALS:
        try:
            os.killpg(pid, sig)
        except ProcessLookupError:
            break
        deadline = time.monotonic() + (grace_s if wait_s is None else wait_s)
        while time.monotonic() < deadline and _group_running(pid):
            time.sleep(0.25)
        if not _group_running(pid):
            break
    if process is not None:
        process.poll()


def _stat(pid: int | str) -> list[str] | None:
    """``/proc/<pid>/stat`` after the command name: state, ppid, pgrp, ..."""
    try:
        return Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
    except (OSError, IndexError):
        return None


def _running(pid: int) -> bool:
    """Whether ``pid`` is still running: it exists and has not exited unreaped (a zombie)."""
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    stat = _stat(pid)
    return stat is None or stat[0] != "Z"


def _group_running(pgid: int) -> bool:
    """Whether any process of group ``pgid`` is still running, zombies aside: an exited
    process stays in its group until its parent, or whoever inherited it, reaps it."""
    try:
        os.killpg(pgid, 0)
    except ProcessLookupError:
        return False
    if not Path("/proc/self/stat").exists():
        return True
    for entry in Path("/proc").iterdir():
        stat = _stat(entry.name) if entry.name.isdigit() else None
        if stat is not None and stat[2] == str(pgid) and stat[0] != "Z":
            return True
    return False


def _summary(plan: Plan) -> str | None:
    if plan.output_dir is None:
        return None
    results = [
        json.loads(line)
        for path in sorted(plan.output_dir.glob("*/episode_results_rank0.jsonl"))
        for line in path.read_text().splitlines()
        if line.strip()
    ]
    if not results:
        return None
    ran = len(results)
    # A window closed or a stop ends a run before the episodes it asked for.
    short = f" ({ran} of {plan.episodes} requested)" if plan.episodes and ran < plan.episodes else ""
    succeeded = sum(1 for result in results if result.get("success") is True)
    judged = sum(1 for result in results if result.get("success") is not None)
    if not plan.judged:
        return f"{ran} episode(s) ran{short}; {plan.mode} episodes are not scored by the scene's success term"
    if not judged:
        return f"{ran} episode(s) ran{short}; the scene has no success term"
    return f"{succeeded}/{ran} episodes succeeded{short}"


def _interrupt(_signum: int, _frame: Any) -> None:
    raise KeyboardInterrupt


def run(plan: Plan) -> int:
    # stop.sh and a plain kill end the run the way Ctrl+C does, so its processes stop too.
    signal.signal(signal.SIGTERM, _interrupt)
    # A launcher started in the background by a script inherits SIGINT ignored, and so would
    # the simulator, which then could not be stopped the quick way. Children start with this
    # process's handlers reset, which takes an ignored SIGINT back to the default.
    signal.signal(signal.SIGINT, signal.default_int_handler)
    plan.run_dir.mkdir(parents=True, exist_ok=True)
    print(f"==> run dir {plan.run_dir}", flush=True)
    for note in plan.notes:
        print(f"==> {note}", flush=True)
    if plan.record is not None:
        plan.record.parent.mkdir(parents=True, exist_ok=True)
        print(f"==> recording {plan.record}", flush=True)
    _decline_telemetry()
    env = _arena_env(plan.env)
    PID_DIR.mkdir(parents=True, exist_ok=True)
    pid_file = PID_DIR / f"{plan.workflow.name}.{os.getpid()}"
    pids: list[int] = []

    def remember(process: subprocess.Popen) -> None:
        pids.append(process.pid)
        pid_file.write_text("".join(f"{pid}\n" for pid in pids))

    server = tool = None
    try:
        if plan.server:
            log = plan.run_dir / "server.log"
            print(f"==> policy server on port {plan.port}; log {log}", flush=True)
            with log.open("w") as handle:
                server = subprocess.Popen(
                    plan.server, cwd=ROOT, env=env, stdout=handle, stderr=subprocess.STDOUT, start_new_session=True
                )
            remember(server)
            _wait_for_port(plan.port, server, log)
        print(f"==> {plan.workflow.name} [{plan.mode}]: {_quote(plan.command)}", flush=True)
        tool = subprocess.Popen(plan.command, cwd=ROOT, env=env, start_new_session=True)
        remember(tool)
        returncode = tool.wait()
    except KeyboardInterrupt:
        returncode = 130
    except RuntimeError as error:
        print(f"!! {error}", file=sys.stderr)
        returncode = 1
    finally:
        # A second Ctrl+C or stop must not cut the cleanup short and leave a server behind.
        signal.signal(signal.SIGINT, signal.SIG_IGN)
        signal.signal(signal.SIGTERM, signal.SIG_IGN)
        for process in (tool, server):
            if process is not None:
                _stop_group(process.pid, process=process)
        pid_file.unlink(missing_ok=True)
    summary = _summary(plan)
    if summary:
        print(f"==> {plan.workflow.name} [{plan.mode}]: {summary}", flush=True)
    # A tool ended by a signal exits as a shell reports it, 128 + the signal.
    return 128 - returncode if returncode < 0 else returncode


def stop(target: str) -> int:
    """Stop the runs this checkout's run.sh started: all, or one workflow's."""
    stopped = 0
    for pid_file in sorted(PID_DIR.glob("*.*")) if PID_DIR.is_dir() else []:
        name, _, launcher = pid_file.name.rpartition(".")
        if target != "all" and name != target:
            continue
        groups = [int(line) for line in pid_file.read_text().split()]
        # The launcher stops what it started, as on Ctrl+C, and reports the episodes so far.
        # Whatever is left after it exits, or if it is gone, is stopped here.
        try:
            os.kill(int(launcher), signal.SIGTERM)
            deadline = time.monotonic() + 60
            while time.monotonic() < deadline and _running(int(launcher)):
                time.sleep(0.5)
        except (ProcessLookupError, ValueError):
            pass
        for pid in reversed(groups):
            _stop_group(pid, grace_s=10.0)
        stopped += len(groups)
        pid_file.unlink(missing_ok=True)
    print(f"stopped {stopped} process group(s)")
    return 0


def list_workflows() -> int:
    for specialty in sorted({workflow.specialty for workflow in load_workflows().values()}):
        print(specialty)
        for workflow in sorted(load_workflows().values(), key=lambda w: w.name):
            if workflow.specialty == specialty:
                print(f"  {workflow.name:32} {workflow.scene:28} {', '.join(workflow.mode_names())}")
    return 0


def main(argv: list[str]) -> int:
    try:
        if not argv or argv[0] in ("-h", "--help"):
            print(__doc__.split("\n\n")[0])
            print(USAGE)
            return 0 if argv else 1
        if argv[0] in RETIRED_COMMANDS:
            raise UsageError(f"{argv[0]} is no longer supported: {RETIRED_COMMANDS[argv[0]]}")
        if argv[0] == "list":
            return list_workflows()
        if argv[0] == "show":
            show(plan(argv[1:]))
            return 0
        if argv[0] == "stop":
            return stop(argv[1] if len(argv) > 1 else "all")
        built = plan(argv)
        if "--dry-run" in argv:
            show(built)
            return 0
        return run(built)
    except UsageError as error:
        print(f"run.sh: {error}", file=sys.stderr)
        return 2


USAGE = """
  ./run.sh list                         workflows, their scenes and modes
  ./run.sh show <workflow> [options]    print the commands a run would execute
  ./run.sh robot-pd <scene> [options]   inspect and tune joint PD drives
  ./run.sh <workflow> [mode] [options]  run it; <workflow> may also be a bare scene
  ./stop.sh [all | <workflow>]          stop the runs run.sh started in this checkout

Modes (choose one; without one, --idle):
  --rule-based              the workflow's rule-based controller
  --policy                  the trained checkpoint, served by its policy stack
  --teleop [DEVICE]         record demonstrations with a teleoperation device
  --replay PATH             replay a recording
  --idle                    open the scene and hold the robot still
  --live                    idle, with Isaac Sim's Python server for live editing
  --mode NAME               any other mode the workflow declares

Options:
  --episodes N              episodes to run, or demonstrations to record
  --episode N               replay only this episode
  --episode-steps N         episode length in control steps
  --record [PATH]           record into PATH (default RUN_DIR/demos.hdf5)
  --run-dir PATH            where outputs, recordings and the server log go
  --checkpoint PATH         serve or load this checkpoint
  --policy-endpoint HOST:PORT  use a policy server that is already running
  --port N                  the policy server's port (default: the stack's, or the next free one)
  --seed N  --device NAME  --presets physx|newton
  --no-cameras              run without the scene's cameras
  --headless                run without the viewport
  --patient-twin PATH       patient-twin manifest for the endoluminal scenes
  --fluoro-backend NAME     synthetic or slang fluoroscopy
  --cloudxr-env PATH        start CloudXR with this environment file for XR teleop
  --dry-run                 print the commands instead of running them
  -- ARGS                   passed on after the environment name
"""


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Minimal liver tissue grasp-and-retract RL example, end to end.

The scripted reference controller demonstrates the full cycle (reach, grasp
the intended liver patch, retract it 20 mm, hold, release). Behaviour cloning
initializes the RSL-RL actor, PPO fine-tunes it in Isaac Lab on the Numerical
Sim liver, and the exported policy runs through the normal ``./run.sh``
launcher. Every stage writes into one run directory; see
``docs/workflows/liver_rl_minimal.md``.

Run from the repository root:

    python3 scripts/liver_rl_minimal.py --headless
    python3 scripts/liver_rl_minimal.py --dry-run
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shlex
import signal
import subprocess
import sys
import time
from contextlib import suppress
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
ARENA_PYTHON = ROOT / "arena" / ".venv" / "bin" / "python"
WORKFLOW = "liver_reach_retract"
TRAIN_WORLDS = 8


@dataclass
class Stage:
    name: str
    summary: str
    command: list[str]
    outputs: list[Path] = field(default_factory=list)
    #: The stage writes its complete outputs before Kit shuts down, so a stuck
    #: shutdown after they exist may be ended (and is recorded as forced).
    outputs_before_shutdown: bool = False
    #: Files or directories a rerun must not find; defaults to ``outputs``.
    replaces: list[Path] = field(default_factory=list)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--run-dir", type=Path, help=f"default: runs/{WORKFLOW}/<YYYYMMDD_HHMMSS>")
    parser.add_argument("--demo-episodes", type=int, default=160, help="reference demonstrations to record")
    parser.add_argument("--action-noise", type=float, default=0.0, help="executed xyz noise while demonstrating")
    parser.add_argument("--bc-epochs", type=int, default=300)
    parser.add_argument("--ppo-iterations", type=int, default=30, help="PPO iterations; the first 5 train the critic")
    parser.add_argument("--eval-episodes", type=int, default=40, help="identical starts for BC and PPO")
    parser.add_argument("--run-episodes", type=int, default=3, help="final ./run.sh --policy episodes")
    parser.add_argument("--headless", action="store_true", help="no viewport for the simulator stages")
    parser.add_argument("--from-stage", help="resume from this stage; earlier outputs must exist")
    parser.add_argument("--dry-run", action="store_true", help="print the stages without running them")
    parser.add_argument(
        "--exit-grace",
        type=float,
        default=120.0,
        help="seconds a simulator stage may take to exit after writing its complete outputs",
    )
    return parser


def stages(args: argparse.Namespace, run_dir: Path) -> list[Stage]:
    headless = ["--headless"] if args.headless else []
    snapshots = run_dir / "snapshots"
    bc = run_dir / "bc"
    experiment = [str(ARENA_PYTHON), "-m", "i4h_rl.liver_experiment"]
    candidates = sorted({u for u in (6, 10, 15, args.ppo_iterations) if 6 <= u <= args.ppo_iterations})
    return [
        Stage(
            "evaluation-starts",
            "freeze separate development and test starts before training",
            [*experiment, "prepare-starts", "--episodes", str(args.eval_episodes), "--output", str(run_dir / "starts")],
            [run_dir / "starts/development.json", run_dir / "starts/test.json"],
        ),
        *(
            Stage(
                f"snapshots-{worlds}",
                f"settle the liver for {worlds} world(s)",
                [
                    str(ARENA_PYTHON),
                    "scripts/prepare_liver_snapshots.py",
                    "--worlds",
                    str(worlds),
                    "--output",
                    str(snapshots),
                    *headless,
                ],
                [snapshots / f"settled_{worlds}.npz", snapshots / f"manifest_{worlds}.json"],
                outputs_before_shutdown=True,
                # The snapshot tool refuses to overwrite either file.
                replaces=[snapshots / f"settled_{worlds}.npz", snapshots / f"manifest_{worlds}.json"],
            )
            for worlds in (TRAIN_WORLDS, 1)
        ),
        Stage(
            "demonstrations",
            "record the reference controller in every world",
            [
                *experiment,
                "collect-demos",
                "--worlds",
                str(TRAIN_WORLDS),
                "--episodes",
                str(args.demo_episodes),
                "--action-noise",
                str(args.action_noise),
                "--output",
                str(bc / "demonstrations.json"),
                *headless,
            ],
            [bc / "demonstrations.npz", bc / "demonstrations.json"],
            outputs_before_shutdown=True,
        ),
        Stage(
            "behaviour-cloning",
            "fit the actor to the demonstrations (no simulator)",
            [
                *experiment,
                "train-bc",
                "--demonstrations",
                str(bc / "demonstrations.npz"),
                "--epochs",
                str(args.bc_epochs),
                "--output",
                str(bc),
            ],
            [bc / "bc.pt", bc / "observations.npz", bc / "bc_training.json"],
        ),
        Stage(
            "import",
            "convert the BC actor into an RSL-RL checkpoint and TorchScript policy",
            [
                *experiment,
                "import-bc",
                "--checkpoint",
                str(bc / "bc.pt"),
                "--observations",
                str(bc / "observations.npz"),
                "--output",
                str(bc / "imported"),
            ],
            [bc / "imported" / name for name in ("native_bc.pt", "policy.pt", "contract.json", "parity.json")],
        ),
        Stage(
            "evaluate-bc",
            "score the BC policy on the fixed starts",
            [
                *experiment,
                "evaluate",
                "--worlds",
                str(TRAIN_WORLDS),
                "--episodes",
                str(args.eval_episodes),
                "--starts",
                str(run_dir / "starts/test.json"),
                "--checkpoint",
                str(bc / "imported" / "policy.pt"),
                "--output",
                str(run_dir / "evaluation" / "bc.json"),
                *headless,
            ],
            [run_dir / "evaluation" / "bc.json"],
            outputs_before_shutdown=True,
        ),
        Stage(
            "ppo",
            "fine-tune the BC actor with PPO in Isaac Lab",
            [
                str(ROOT / "train.sh"),
                "rl",
                WORKFLOW,
                "--checkpoint",
                str(bc / "imported" / "native_bc.pt"),
                "--num-envs",
                str(TRAIN_WORLDS),
                "--epochs",
                str(args.ppo_iterations),
                "--run-dir",
                str(run_dir / "ppo"),
                "--set",
                "agent.algorithm.class_name=i4h_rl.liver_ppo:MinimalLiverPPO",
                "--set",
                "agent.algorithm.schedule=fixed",
            ],
            [run_dir / "ppo" / "model_final.pt", run_dir / "ppo" / "run.json"],
            replaces=[run_dir / "ppo"],
        ),
        Stage(
            "select-ppo",
            "select the earliest best checkpoint on development starts",
            [
                *experiment,
                "select-ppo",
                "--worlds",
                str(TRAIN_WORLDS),
                "--checkpoint",
                str(run_dir / "ppo"),
                "--starts",
                str(run_dir / "starts/development.json"),
                "--candidate-updates",
                *map(str, candidates),
                "--output",
                str(run_dir / "selection/selection.json"),
                *headless,
            ],
            [run_dir / "selection/selection.json", run_dir / "selection/model_selected.pt"],
            outputs_before_shutdown=True,
        ),
        Stage(
            "export",
            "export the PPO actor as policy.pt and contract.json",
            [
                str(ROOT / "train.sh"),
                "rl",
                "export",
                WORKFLOW,
                "--checkpoint",
                str(run_dir / "selection" / "model_selected.pt"),
                "--output-dir",
                str(run_dir / "export"),
            ],
            [run_dir / "export" / "policy.pt", run_dir / "export" / "contract.json"],
            # Export requires an absent or empty output directory.
            replaces=[run_dir / "export"],
        ),
        Stage(
            "evaluate-ppo",
            "score the PPO policy on the same starts",
            [
                *experiment,
                "evaluate",
                "--worlds",
                str(TRAIN_WORLDS),
                "--episodes",
                str(args.eval_episodes),
                "--starts",
                str(run_dir / "starts/test.json"),
                "--checkpoint",
                str(run_dir / "export" / "policy.pt"),
                "--output",
                str(run_dir / "evaluation" / "ppo.json"),
                *headless,
            ],
            [run_dir / "evaluation" / "ppo.json"],
            outputs_before_shutdown=True,
        ),
        Stage(
            "run",
            "run the exported policy through the normal workflow launcher",
            [
                str(ROOT / "run.sh"),
                WORKFLOW,
                "--policy",
                "--checkpoint",
                str(run_dir / "export"),
                "--episodes",
                str(args.run_episodes),
                "--record",
                "--run-dir",
                str(run_dir / "run"),
                *headless,
            ],
            [run_dir / "run" / "demos.hdf5"],
            replaces=[run_dir / "run"],
        ),
    ]


def _environment(run_dir: Path) -> dict[str, str]:
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env["I4H_LIVER_SNAPSHOTS"] = str(run_dir / "snapshots")
    env["I4H_LIVER_OBJECTIVE"] = "full_cycle"
    env.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    roots = [str(ROOT / "rl"), str(ROOT / "common")]
    env["PYTHONPATH"] = os.pathsep.join([*roots, *([env["PYTHONPATH"]] if env.get("PYTHONPATH") else [])])
    return env


def _missing_dependencies() -> list[str]:
    from_pyproject = None
    try:
        sys.path.insert(0, str(ROOT / "common"))
        from i4h_common.paths import numerical_source

        from_pyproject = numerical_source(ROOT)
    except Exception as error:  # noqa: BLE001 - reported as a setup problem
        return [f"cannot resolve the Numerical Sim source from arena/pyproject.toml: {error}"]
    missing = []
    if not ARENA_PYTHON.is_file():
        missing.append(f"Arena environment {ARENA_PYTHON}")
    if not from_pyproject.is_dir():
        missing.append(
            f"Numerical Sim source {from_pyproject} (provided by the pinned "
            "isaac-for-healthcare/i4h-physics-simulation repository; see third_party/setup.sh)"
        )
    return missing


def _set_aside(stage: Stage, stamp: str) -> list[str]:
    """Move a rerun stage's previous outputs aside instead of deleting them."""
    moved = []
    for path in stage.replaces or stage.outputs:
        if path.exists():
            target = path.with_name(f"{path.name}.replaced-{stamp}")
            path.rename(target)
            moved.append(str(target))
    return moved


def _stop_process_group(process):
    """Stop only the session created for this stage, including launcher children."""
    with suppress(ProcessLookupError):
        os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        pass
    finally:
        # A launcher may exit before its simulator. Kill remaining group members.
        with suppress(ProcessLookupError):
            os.killpg(process.pid, signal.SIGKILL)
        process.wait()


def _execute(stage: Stage, env: dict[str, str], log: Path, exit_grace: float) -> tuple[int, bool]:
    """Run one stage; end a stuck Kit shutdown once this run's complete outputs exist."""
    started = time.time()

    def written() -> bool:
        return all(path.is_file() and path.stat().st_mtime >= started for path in stage.outputs)

    with log.open("w") as stream:
        process = subprocess.Popen(
            stage.command, cwd=ROOT, env=env, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True
        )
        try:
            written_at = None
            while (returncode := process.poll()) is None:
                if stage.outputs_before_shutdown and written():
                    written_at = written_at or time.monotonic()
                    if time.monotonic() - written_at > exit_grace:
                        stream.write(f"\n[liver_rl_minimal] outputs complete; ending shutdown after {exit_grace} s\n")
                        stream.flush()
                        return 0, True
                time.sleep(1)
        finally:
            _stop_process_group(process)
    return returncode, False


def _score(path: Path) -> dict | None:
    if not path.is_file():
        return None
    report = json.loads(path.read_text())
    return {key: report[key] for key in ("episodes", "successes", "success_rate", "wilson_95", "failures")}


def _evaluations(run_dir, records):
    return {
        name: _score(run_dir / "evaluation" / f"{name}.json")
        if records.get(f"evaluate-{name}", {}).get("returncode") == 0
        and not records[f"evaluate-{name}"].get("missing_outputs")
        else None
        for name in ("bc", "ppo")
    }


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.ppo_iterations < 6 or min(args.demo_episodes, args.bc_epochs, args.eval_episodes, args.run_episodes) < 1:
        print("require at least 6 PPO iterations and positive episode/epoch counts", file=sys.stderr)
        return 2
    run_dir = (args.run_dir or ROOT / "runs" / WORKFLOW / datetime.now().strftime("%Y%m%d_%H%M%S")).resolve()
    plan = stages(args, run_dir)
    names = [stage.name for stage in plan]
    if args.from_stage and args.from_stage not in names:
        print(f"unknown stage {args.from_stage!r}; stages: {', '.join(names)}", file=sys.stderr)
        return 2
    start = names.index(args.from_stage) if args.from_stage else 0
    if args.dry_run:
        print(f"# run dir {run_dir}")
        for index, stage in enumerate(plan):
            marker = "skip" if index < start else "run "
            print(f"# [{marker}] {stage.name}: {stage.summary}")
            print(shlex.join(stage.command))
        return 0
    missing = _missing_dependencies()
    if missing:
        print("missing dependencies; run ./setup.sh:\n  " + "\n  ".join(missing), file=sys.stderr)
        return 2
    for stage in plan[:start]:
        absent = [str(path) for path in stage.outputs if not path.is_file()]
        if absent:
            print(f"cannot resume at {args.from_stage}: {stage.name} outputs are missing: {absent}", file=sys.stderr)
            return 2
    summary_path = run_dir / "summary.json"
    summary = json.loads(summary_path.read_text()) if start and summary_path.is_file() else {}
    records = {record["name"]: record for record in summary.get("stages", [])}
    for stage in plan[:start]:
        record = records.get(stage.name, {})
        expected = record.get("outputs", {})
        if (
            record.get("returncode") != 0
            or record.get("command") != stage.command
            or any(expected.get(str(path)) != _sha256(path) for path in stage.outputs)
        ):
            print(
                f"cannot resume: {stage.name} has changed or unverified outputs/options; "
                f"rerun with --from-stage {stage.name}",
                file=sys.stderr,
            )
            return 2
    # Prior downstream records and scores no longer describe the rerun's inputs.
    records = {stage.name: records[stage.name] for stage in plan[:start]}
    run_dir.mkdir(parents=True, exist_ok=True)
    (run_dir / "logs").mkdir(exist_ok=True)
    summary.update(run_dir=str(run_dir), arguments={k: str(v) for k, v in vars(args).items()})
    summary["stages"] = list(records.values())
    summary["evaluation"] = _evaluations(run_dir, records)
    summary_path.write_text(json.dumps(summary, indent=2) + "\n")
    env = _environment(run_dir)
    print(f"==> run dir {run_dir}", flush=True)
    status = 0
    for stage in plan[start:]:
        log = run_dir / "logs" / f"{stage.name}.log"
        print(f"==> {stage.name}: {stage.summary}\n    {shlex.join(stage.command)}\n    log {log}", flush=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        set_aside = _set_aside(stage, stamp)
        if log.exists():
            previous_log = log.with_name(f"{stage.name}.replaced-{stamp}.log")
            log.rename(previous_log)
            set_aside.append(str(previous_log))
        for path in set_aside:
            print(f"    previous output kept as {path}", flush=True)
        began = time.monotonic()
        returncode, forced_exit = _execute(stage, env, log, args.exit_grace)
        absent = [str(path) for path in stage.outputs if not path.is_file()]
        records[stage.name] = dict(
            name=stage.name,
            command=stage.command,
            returncode=returncode,
            forced_exit=forced_exit,
            set_aside=set_aside,
            seconds=round(time.monotonic() - began, 1),
            log=str(log),
            outputs={str(path): _sha256(path) for path in stage.outputs if path.is_file()},
            missing_outputs=absent,
        )
        summary["stages"] = [records[name] for name in names if name in records]
        summary["evaluation"] = _evaluations(run_dir, records)
        summary_path.write_text(json.dumps(summary, indent=2) + "\n")
        if returncode != 0 or absent:
            print(f"!! {stage.name} failed (exit {returncode}, missing {absent}); see {log}", file=sys.stderr)
            print(f"   resume with: --run-dir {run_dir} --from-stage {stage.name}", file=sys.stderr)
            status = returncode or 1
            break
        print(f"    done in {records[stage.name]['seconds']} s", flush=True)
    for name, score in summary["evaluation"].items():
        if score:
            print(f"==> {name}: {score['successes']}/{score['episodes']} full cycles", flush=True)
    print(f"==> summary {summary_path}", flush=True)
    return status


if __name__ == "__main__":
    # Turn SIGTERM into stack unwinding so the active stage's finally block runs.
    def _terminate(signum, frame):
        raise SystemExit(128 + signum)

    signal.signal(signal.SIGTERM, _terminate)
    sys.exit(main())

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run repository test suites or GPU workflow smokes."""

from __future__ import annotations

import argparse
import os
import re
import shutil
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


@dataclass(frozen=True)
class TestCommand:
    name: str
    project: str
    paths: tuple[str, ...]
    project_has_dev_extra: bool = True
    # Projects that depend on an internal component checkout cannot even resolve their
    # environment before `./setup.sh` has cloned it, so they are skipped rather than run
    # and reported as a failure.
    requires_checkout: str | None = None

    def missing_checkout(self) -> Path | None:
        if self.requires_checkout is None:
            return None
        path = ROOT / self.requires_checkout
        return None if path.is_dir() else path

    def argv(self, *, coverage: bool = False) -> list[str]:
        argv = [
            "uv",
            "run",
            "--project",
            self.project,
        ]
        argv.extend(("--extra", "dev") if self.project_has_dev_extra else ("--with", "pytest>=8.0"))
        if coverage:
            return [
                *argv,
                "--with",
                f"coverage=={COVERAGE_VERSION}",
                "coverage",
                "run",
                "--parallel-mode",
                f"--rcfile={COVERAGE_CONFIG}",
                "-m",
                "pytest",
                "-ra",
                *self.paths,
            ]
        return [*argv, "pytest", "-ra", *self.paths]


SUITES = {
    "light": (
        TestCommand(
            "workflow contracts",
            "workflows",
            (
                "tests",
                "common/tests",
                "engine/tests",
                "workflows/tests",
                "tasks/basic/tests",
                "tasks/gr00t_n15/tests",
                "tasks/gr00t_n16/tests",
                "tasks/ik/tests",
                "tasks/rsl_rl/tests",
                "tasks/teleop/tests",
            ),
        ),
    ),
    "tools": (
        TestCommand("trajectory mimic", "tools/mimic", ("tools/mimic/tests",)),
        TestCommand("dataset conversion", "tools/dataset", ("tools/dataset/tests",)),
        TestCommand(
            "patient twin",
            "tools/patient_twin",
            ("tools/patient_twin/tests",),
            requires_checkout="third_party/i4h-digital-twin-internal/patient-digital-twin/vasculature_digital_twin",
        ),
    ),
    "arena": (
        TestCommand(
            "IsaacLab Arena",
            "arena",
            ("arena/tests",),
            project_has_dev_extra=False,
            requires_checkout="third_party/IsaacLab-Arena-0a1b8c2",
        ),
    ),
}

COVERAGE_VERSION = "7.4.4"
COVERAGE_CONFIG = ROOT / ".coveragerc"
COVERAGE_XML = ROOT / "coverage.xml"
GPU_RECORDING = "verify.hdf5"
GPU_EPISODE_RESULT = re.compile(r"\b[01]/1 episodes succeeded \([1-9]\d* attempts\)")


@dataclass(frozen=True)
class WorkflowSmoke:
    workflow: str
    mode_args: tuple[str, ...]


GPU_SMOKES = (
    WorkflowSmoke("surgical_reach_psm", ("--rule-based",)),
    WorkflowSmoke("assemble_trocar", ("--policy",)),
    WorkflowSmoke("ultrasound_liver_scan", ("--rule-based",)),
    WorkflowSmoke("scissor_pick_and_place", ("--rule-based",)),
    WorkflowSmoke("endoluminal_navigation", ("--mode", "demo")),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("ci", "all", "gpu", *SUITES),
        default="ci",
        help="test group to run (gpu runs only the recorded simulator smokes)",
    )
    parser.add_argument("--timeout", type=int, default=1200, help="timeout for each pytest command")
    parser.add_argument(
        "--coverage",
        action="store_true",
        help="collect CPU-suite coverage and write coverage.xml (not valid with --suite gpu)",
    )
    return parser


def _run_command(argv: list[str], *, timeout: int, capture: bool = False) -> subprocess.CompletedProcess[str] | None:
    print("    " + " ".join(argv), flush=True)
    try:
        result = subprocess.run(
            argv,
            cwd=ROOT,
            timeout=timeout,
            check=False,
            text=True,
            stdout=subprocess.PIPE if capture else None,
            stderr=subprocess.STDOUT if capture else None,
            env={**os.environ, "OMNI_KIT_ACCEPT_EULA": "Y", "PRIVACY_CONSENT": "Y"},
        )
    except subprocess.TimeoutExpired as exc:
        if exc.stdout:
            print(exc.stdout.decode() if isinstance(exc.stdout, bytes) else exc.stdout, end="")
        print(f"!! timed out after {timeout}s", file=sys.stderr)
        return None
    if capture and result.stdout:
        print(result.stdout, end="")
    return result


def _clear_coverage_artifacts() -> None:
    for path in (COVERAGE_XML, *ROOT.glob(".coverage"), *ROOT.glob(".coverage.*")):
        path.unlink(missing_ok=True)


def _coverage_argv(*args: str) -> list[str]:
    return [
        "uv",
        "run",
        "--project",
        "workflows",
        "--with",
        f"coverage=={COVERAGE_VERSION}",
        "coverage",
        *args,
    ]


def _finish_coverage(timeout: int) -> bool:
    print("\n==> [coverage] combine CPU suite data", flush=True)
    for args in (("combine",), ("report", "--show-missing"), ("xml", "-o", str(COVERAGE_XML))):
        result = _run_command(_coverage_argv(*args), timeout=timeout)
        if result is None or result.returncode:
            print(f"!! coverage {' '.join(args)} failed", file=sys.stderr)
            return False
    return True


def _run_gpu_smoke(smoke: WorkflowSmoke, timeout: int) -> bool:
    mode = " ".join(smoke.mode_args)
    print(f"\n==> [gpu] {smoke.workflow} {mode}", flush=True)
    rollout = _run_command(
        [
            "./run.sh",
            smoke.workflow,
            *smoke.mode_args,
            "--episodes",
            "1",
            "--attempts",
            "1",
            "--seed",
            "0",
            "--headless",
            "--record",
            GPU_RECORDING,
            "--record-failures",
        ],
        timeout=timeout,
        capture=True,
    )
    if rollout is None:
        return False

    output = rollout.stdout or ""
    if GPU_EPISODE_RESULT.search(output) is None:
        print("!! rollout did not report a completed episode", file=sys.stderr)
        return False
    run_dir_match = re.search(r"^==> run dir (.+)$", output, flags=re.MULTILINE)
    if run_dir_match is None:
        print("!! rollout did not report its run directory", file=sys.stderr)
        return False

    recording = Path(run_dir_match.group(1)) / GPU_RECORDING
    if not recording.is_file():
        print(f"!! rollout recording is missing: {recording}", file=sys.stderr)
        return False

    print("\n==> [gpu] inspect rollout recording", flush=True)
    inspection = _run_command(
        ["uv", "run", "--project", "tools/dataset", "i4h-dataset", "inspect", str(recording), "--segments"],
        timeout=timeout,
    )
    if inspection is None or inspection.returncode:
        print("!! rollout recording inspection failed", file=sys.stderr)
        return False
    return True


def main() -> int:
    args = _parser().parse_args()
    if shutil.which("uv") is None:
        print("uv is required to run the independently locked test projects", file=sys.stderr)
        return 2
    if args.coverage and args.suite == "gpu":
        print("--coverage applies to CPU pytest suites, not GPU workflow smokes", file=sys.stderr)
        return 2
    if args.coverage:
        _clear_coverage_artifacts()
    if args.suite == "ci":
        selected = ("light", "tools")
    elif args.suite == "all":
        selected = tuple(SUITES)
    elif args.suite == "gpu":
        selected = ()
    else:
        selected = (args.suite,)
    failures: list[str] = []
    skipped: list[str] = []
    for suite in selected:
        for command in SUITES[suite]:
            print(f"\n==> [{suite}] {command.name}", flush=True)
            missing = command.missing_checkout()
            if missing is not None:
                print(f"    skipped: {missing.relative_to(ROOT)} is missing; run ./setup.sh to clone it")
                skipped.append(command.name)
                continue
            result = _run_command(command.argv(coverage=args.coverage), timeout=args.timeout)
            if result is None or result.returncode:
                failures.append(command.name)
    if args.coverage and not _finish_coverage(args.timeout):
        failures.append("coverage report")
    if args.suite == "gpu":
        for smoke in GPU_SMOKES:
            if not _run_gpu_smoke(smoke, args.timeout):
                failures.append(smoke.workflow)
    if skipped:
        print("\nskipped, no component checkout: " + ", ".join(skipped))
    if failures:
        print("\nfailed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nall selected test suites passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

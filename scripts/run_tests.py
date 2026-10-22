#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Run repository test suites or GPU environment smokes."""

from __future__ import annotations

import argparse
import os
import shutil
import subprocess
import sys
import tempfile
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
    pytest_args: tuple[str, ...] = ()

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
                *self.pytest_args,
                *self.paths,
            ]
        return [*argv, "pytest", "-ra", *self.pytest_args, *self.paths]


SUITES = {
    "light": (
        TestCommand(
            "contracts and manifests",
            "common",
            (
                "tests",
                "common/tests",
                "tasks/gr00t_n15/tests",
                "tasks/gr00t_n16/tests",
                "tasks/gr00t_n17/tests/test_serve.py",
                # The openpi-free half of the PI0 trainer tests; the rest need
                # the openpi checkout and run in that project's environment.
                "tasks/openpi_pi0/tests/test_openpi_pi0_train_options.py",
                "rl/tests/test_rsl_rl_arm_runtime.py",
            ),
        ),
    ),
    "tools": (
        TestCommand("trajectory mimic", "tools/mimic", ("tools/mimic/tests",)),
        TestCommand("VLM annotator", "tools/annotator", ("tools/annotator/tests",)),
        TestCommand("dataset conversion", "tools/dataset", ("tools/dataset/tests",)),
        TestCommand(
            "patient twin",
            "tools/patient_twin",
            ("tools/patient_twin/tests",),
            requires_checkout="third_party/i4h-digital-twin-internal/patient-digital-twin/vasculature_digital_twin",
        ),
    ),
    "rl": (
        TestCommand(
            "RL CPU regressions",
            "rl",
            ("rl/tests",),
            requires_checkout="third_party/RLinf-a4b6abe",
            pytest_args=("-p", "i4h_rl.pytest_ci"),
        ),
    ),
    "arena": (
        TestCommand(
            "IsaacLab Arena",
            "arena",
            ("arena/tests",),
            project_has_dev_extra=False,
            requires_checkout="third_party/IsaacLab-Arena-c7edf3a",
        ),
    ),
}

COVERAGE_VERSION = "7.4.4"
COVERAGE_CONFIG = ROOT / ".coveragerc"
COVERAGE_XML = ROOT / "coverage.xml"
GPU_RECORDING = "verify.hdf5"


@dataclass(frozen=True)
class WorkflowSmoke:
    """One recorded, headless episode of a workflow mode, run through run.sh."""

    workflow: str
    mode: str


GPU_SMOKES = (
    WorkflowSmoke("surgical_reach_psm", "rule-based"),
    WorkflowSmoke("surgical_lift_needle", "rule-based"),
    # run.sh serves the checkpoint and starts the rollout once the server listens.
    WorkflowSmoke("assemble_trocar", "policy"),
    WorkflowSmoke("ultrasound_liver_scan", "rule-based"),
    WorkflowSmoke("scissor_pick_and_place", "rule-based"),
    WorkflowSmoke("endoluminal_navigation", "demo"),
)


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--suite",
        choices=("ci", "all", "gpu", "dependency-pins", *SUITES),
        default="ci",
        help="test group to run (gpu runs recorded simulator smokes and dependency-pin checks)",
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
        "common",
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


def _run_workflow_smoke(smoke: WorkflowSmoke, timeout: int) -> bool:
    print(f"\n==> [gpu] {smoke.workflow} {smoke.mode}", flush=True)
    run_dir = Path(tempfile.mkdtemp(prefix=f"i4h-smoke-{smoke.workflow}-"))
    argv = [
        str(ROOT / "run.sh"),
        smoke.workflow,
        "--mode",
        smoke.mode,
        "--episodes",
        "1",
        "--seed",
        "0",
        "--headless",
        "--run-dir",
        str(run_dir),
        "--record",
        str(run_dir / GPU_RECORDING),
    ]
    print("    " + " ".join(argv), flush=True)
    process = subprocess.Popen(argv, cwd=ROOT)
    try:
        returncode = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        # run.sh stops its simulator and policy server on SIGTERM, as on Ctrl+C.
        process.terminate()
        process.wait()
        print(f"!! timed out after {timeout}s", file=sys.stderr)
        return False
    if returncode:
        print("!! run.sh failed", file=sys.stderr)
        return False
    results = sorted(run_dir.glob("outputs/*/episode_results_rank0.jsonl"))
    if not results or not results[-1].read_text().strip():
        print("!! the policy runner reported no completed episode", file=sys.stderr)
        return False
    recording = run_dir / GPU_RECORDING
    if not recording.is_file():
        print(f"!! rollout recording is missing: {recording}", file=sys.stderr)
        return False
    return True


def main() -> int:
    args = _parser().parse_args()
    if shutil.which("uv") is None:
        print("uv is required to run the independently locked test projects", file=sys.stderr)
        return 2
    if args.coverage and args.suite in {"gpu", "dependency-pins"}:
        print("--coverage applies to CPU pytest suites, not GPU environment smokes", file=sys.stderr)
        return 2
    if args.coverage:
        _clear_coverage_artifacts()
    if args.suite == "ci":
        selected = ("light", "tools", "rl")
    elif args.suite == "all":
        selected = tuple(SUITES)
    elif args.suite in {"gpu", "dependency-pins"}:
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
                if suite == "rl" and os.environ.get("CI"):
                    failures.append(command.name)
                skipped.append(command.name)
                continue
            result = _run_command(command.argv(coverage=args.coverage), timeout=args.timeout)
            if result is None or result.returncode:
                failures.append(command.name)
    if args.coverage and not _finish_coverage(args.timeout):
        failures.append("coverage report")
    if args.suite == "gpu":
        for smoke in GPU_SMOKES:
            if not _run_workflow_smoke(smoke, args.timeout):
                failures.append(f"{smoke.workflow} {smoke.mode}")
    if args.suite in {"gpu", "dependency-pins"}:
        result = _run_command(
            [
                "uv",
                "run",
                "--project",
                "arena",
                "--with",
                "pytest>=8.0",
                "python",
                "scripts/validate_dependency_pins.py",
                "--timeout",
                str(args.timeout),
            ],
            timeout=2 * args.timeout + 30,
        )
        if result is None or result.returncode:
            failures.append("in-Kit CUDA and Vulkan dependency pins")
    if skipped:
        print("\nskipped, no component checkout: " + ", ".join(skipped))
    if failures:
        print("\nfailed: " + ", ".join(failures), file=sys.stderr)
        return 1
    print("\nall selected test suites passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

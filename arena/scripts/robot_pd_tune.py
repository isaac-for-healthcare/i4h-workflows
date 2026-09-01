# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""One-command wrapper for live robot PD diagnostics."""

from __future__ import annotations

import argparse
import json
import os
import signal
import socket
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    values = [item.strip() for item in value.split(",") if item.strip()]
    return values or None


def _deep_update(dst: dict[str, Any], src: dict[str, Any]) -> dict[str, Any]:
    for key, value in src.items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="./run.sh robot-pd",
        description="Run robot PD diagnostics through a live workflow session.",
    )
    parser.add_argument("workflow", help="Workflow whose scene contains the robot articulation.")
    parser.add_argument(
        "--modes",
        default="all",
        help="Comma-separated modes: all, inspect-usd, direct-state, step-response, trajectory.",
    )
    parser.add_argument("--joint-names", default=None, help="Comma-separated exact joint names to tune.")
    parser.add_argument("--joint-names-expr", default=None, help="Comma-separated regex patterns for joint names.")
    parser.add_argument("--plots", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--sample-every", type=int, default=None, help="Record every Nth sample for dynamic modes.")
    parser.add_argument("--config", type=Path, default=None, help="Optional JSON config merged into generated config.")
    parser.add_argument("--sim-log", type=Path, default=None, help="Live simulator log path.")
    parser.add_argument("--timeout", type=float, default=180.0, help="Python-server ready timeout in seconds.")
    parser.add_argument(
        "--script-timeout",
        type=float,
        default=1800.0,
        help="Remote diagnostic execution timeout in seconds.",
    )
    parser.add_argument("--headless", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--disable-cameras",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Disable rendering support only when the selected Scene declares no cameras.",
    )
    parser.add_argument("--keep-live", action="store_true", help="Leave the live simulator running after diagnostics.")
    parser.add_argument("--print-json", action="store_true", help="Print the full Python-server JSON response.")
    parser.add_argument("--dry-run", action="store_true", help="Print config and commands without launching Isaac.")
    return parser


def _generated_config(args: argparse.Namespace) -> dict[str, Any]:
    common: dict[str, Any] = {"plots": {"enabled": bool(args.plots)}}
    if names := _csv(args.joint_names):
        common["joint_names"] = names
    if exprs := _csv(args.joint_names_expr):
        common["joint_names_expr"] = exprs

    config: dict[str, Any] = {
        "pd_diagnostics": {"modes": _csv(args.modes) or ["all"]},
        "common": common,
    }
    if args.sample_every is not None:
        config["joint_step_response"] = {"sample_every": args.sample_every}
        config["joint_trajectory_response"] = {"sample_every": args.sample_every}
    if args.config is not None:
        _deep_update(config, json.loads(args.config.read_text(encoding="utf-8")))
    return config


def _log(message: str) -> None:
    print(message, flush=True)


def _run_remote_script(
    remote_client: Path,
    script_path: Path,
    *,
    timeout_s: float,
) -> dict[str, Any]:
    client_timeout = max(timeout_s + 30.0, 60.0)
    command = [
        sys.executable,
        str(remote_client),
        "--raw",
        "--timeout",
        str(client_timeout),
        "--execution-timeout",
        str(timeout_s),
        "--args-json",
        json.dumps({"script_path": str(script_path)}),
        "--file",
        str(script_path),
    ]
    completed = subprocess.run(command, check=False, text=True, capture_output=True)
    try:
        outer = json.loads(completed.stdout)
    except json.JSONDecodeError as exc:
        detail = completed.stderr.strip() or completed.stdout.strip() or "no Python-server response"
        return {"ok": False, "error": f"could not parse Python-server response: {exc}: {detail}"}
    if outer.get("status") != "ok":
        error = outer.get("evalue") or outer.get("result") or outer
        return {"ok": False, "error": str(error)}
    output = str(outer.get("output", "")).strip()
    if not output:
        return {"ok": False, "error": "PD diagnostics produced no JSON output"}
    try:
        result = json.loads(output)
    except json.JSONDecodeError as exc:
        return {"ok": False, "error": f"PD diagnostics returned invalid JSON: {exc}"}
    return {"ok": True, "result": result}


def _remote_client(workflows_root: Path) -> Path:
    matches = sorted(
        (workflows_root / "third_party").glob("IsaacSim-*/skills/isaac-sim-remote/scripts/isaacsim_send.py")
    )
    if len(matches) != 1:
        raise RuntimeError(f"expected one pinned Isaac Sim remote client; found {len(matches)}")
    return matches[0]


def _wait_for_runner(
    process: subprocess.Popen[str],
    remote_client: Path,
    workflow: str,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        if process.poll() is not None:
            raise RuntimeError(f"live simulator exited before its runner was ready (code {process.returncode})")
        try:
            with socket.create_connection(("127.0.0.1", 8226), timeout=1.0):
                completed = subprocess.run(
                    [
                        sys.executable,
                        str(remote_client),
                        "--raw",
                        "--timeout",
                        "5",
                        "from i4h_arena.runner import active_runner; print(active_runner().workflow.name)",
                    ],
                    check=False,
                    capture_output=True,
                    text=True,
                    timeout=10,
                )
                if completed.returncode == 0:
                    response = json.loads(completed.stdout)
                    if response.get("status") == "ok" and str(response.get("output", "")).strip() == workflow:
                        return
        except OSError:
            pass
        except (json.JSONDecodeError, subprocess.TimeoutExpired):
            pass
        time.sleep(1.0)
    raise TimeoutError(f"workflow {workflow!r} did not expose an active runner within {timeout_s:.0f}s")


def _stop_process(process: subprocess.Popen[str]) -> None:
    if process.poll() is not None:
        return
    os.killpg(process.pid, signal.SIGTERM)
    try:
        process.wait(timeout=30)
    except subprocess.TimeoutExpired:
        os.killpg(process.pid, signal.SIGKILL)
        process.wait(timeout=10)


def _as_dict(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _print_response_summary(response: dict[str, Any], run_dir_hint: Path) -> None:
    result = _as_dict(response.get("result"))
    json_path = _as_dict(result.get("outputs")).get("json") or (run_dir_hint / "robot_pd_diagnostics.json")
    run_dir = Path(json_path).parent
    status = "completed" if response.get("ok") else "failed"
    _log(f"[robot-pd] diagnostics {status}")
    if error := response.get("error"):
        _log(f"[robot-pd] error: {error}")
    if not result:
        return

    modes = result.get("modes") or []
    mode_text = ", ".join(str(mode) for mode in modes) if modes else "unknown"
    _log(f"[robot-pd] env: {result.get('env_id', 'unknown')}")
    _log(f"[robot-pd] modes: {mode_text}")
    _log(f"[robot-pd] results: {run_dir}")

    reports = _as_dict(result.get("results"))
    step_report = _as_dict(reports.get("step-response"))
    joint_count = len(step_report.get("joint_order") or [])
    if joint_count:
        _log(f"[robot-pd] joints tested: {joint_count}")

    plot_dirs = _plot_dirs(result)
    if plot_dirs:
        _log("[robot-pd] plots:")
        for label, path in plot_dirs:
            _log(f"  - {label}: {path}")

    recommendations = _changed_recommendations(step_report)
    if recommendations:
        _log("[robot-pd] tuning notes:")
        for joint, reasons in recommendations:
            reason_text = "; ".join(reasons) if reasons else "gain change recommended"
            _log(f"  - {joint}: {reason_text}")
    elif step_report:
        _log("[robot-pd] tuning notes: no gain changes recommended")


def _plot_dirs(result: dict[str, Any]) -> list[tuple[str, str]]:
    mode_outputs = _as_dict(_as_dict(result.get("outputs")).get("mode_outputs"))
    plot_dirs = []
    for mode, outputs in mode_outputs.items():
        plots = _as_dict(_as_dict(outputs).get("plots"))
        if plots.get("enabled") and plots.get("output_dir"):
            plot_dirs.append((str(mode), str(plots["output_dir"])))
    return plot_dirs


def _changed_recommendations(step_report: dict[str, Any]) -> list[tuple[str, list[str]]]:
    recommendations = []
    for joint, joint_result in _as_dict(step_report.get("results")).items():
        recommendation = _as_dict(_as_dict(joint_result).get("recommendation"))
        if recommendation.get("changed"):
            reasons = recommendation.get("reasons") or []
            recommendations.append((str(joint), [str(reason) for reason in reasons]))
    return recommendations


def main(argv: list[str] | None = None) -> int:
    args, passthrough = _parser().parse_known_args(argv)
    arena_dir = Path(__file__).resolve().parents[1]
    workflows_root = arena_dir.parent
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = workflows_root / "runs" / args.workflow / f"{timestamp}_pd_tuning"
    config_path = run_dir / "robot_pd_tuning_config.json"
    sim_log = args.sim_log or (run_dir / "sim.log")
    script_path = arena_dir / "scripts" / "robot_pd_diagnostics.py"
    remote_client = _remote_client(workflows_root)
    config = _generated_config(args)
    run_command = [
        str(workflows_root / "run.sh"),
        args.workflow,
        "--live",
        "--run-dir",
        str(run_dir),
    ]
    if args.headless:
        run_command.append("--headless")
    if args.disable_cameras:
        run_command.append("--no-cameras")
    run_command.extend(passthrough)
    remote_command = [
        sys.executable,
        str(remote_client),
        "--raw",
        "--timeout",
        str(max(args.script_timeout + 30.0, 60.0)),
        "--execution-timeout",
        str(args.script_timeout),
        "--args-json",
        json.dumps({"script_path": str(arena_dir / "scripts" / "robot_pd_diagnostics.py")}),
        "--file",
        str(script_path),
    ]

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_path": str(config_path),
                    "run_dir": str(run_dir),
                    "config": config,
                    "live_command": run_command,
                    "remote_command": remote_command,
                    "diagnostics_script": str(script_path),
                    "script_timeout_s": args.script_timeout,
                    "print_json": args.print_json,
                },
                indent=2,
            )
        )
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    sim_log.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _log(f"[robot-pd] wrote config: {config_path}")
    environment = os.environ.copy()
    environment["I4H_PD_RUN_DIR"] = str(run_dir)
    with sim_log.open("w", encoding="utf-8") as log_handle:
        process = subprocess.Popen(
            run_command,
            cwd=workflows_root,
            env=environment,
            stdout=log_handle,
            stderr=subprocess.STDOUT,
            text=True,
            start_new_session=True,
        )
    try:
        _log(f"[robot-pd] waiting for live workflow={args.workflow}; log={sim_log}")
        _wait_for_runner(process, remote_client, args.workflow, args.timeout)
        _log(f"[robot-pd] running diagnostics through the Python server for workflow={args.workflow}")
        response = _run_remote_script(remote_client, script_path, timeout_s=args.script_timeout)
        _print_response_summary(response, run_dir)
        if args.print_json:
            print(json.dumps(response, indent=2), flush=True)
        ok = bool(response.get("ok"))
    finally:
        if args.keep_live:
            _log(f"[robot-pd] live session left running; stop with: {workflows_root / 'stop.sh'} all")
        else:
            _log(f"[robot-pd] stopping live workflow={args.workflow}")
            _stop_process(process)

    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

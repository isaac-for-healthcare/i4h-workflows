# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Robot PD diagnostics on a scene's robot, as an Isaac Lab script.

Builds the scene's IsaacLab-Arena environment in this process, then steps its
robot through the requested diagnostics::

    arena/.venv/bin/python arena/scripts/robot_pd_tune.py psm_reach --modes inspect-usd,step-response

Options this script does not know, such as ``--presets newton`` or ``--viz kit``,
go to Isaac Lab's app launcher and Arena's environment builder.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
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
        prog="robot_pd_tune.py",
        description="Run robot PD diagnostics on a scene's robot articulation.",
    )
    parser.add_argument("scene", help="Scene whose environment contains the robot articulation.")
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
    parser.add_argument("--print-json", action="store_true", help="Print the full diagnostics JSON.")
    parser.add_argument("--dry-run", action="store_true", help="Print config and arguments without launching Isaac.")
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


def _environment_argv(scene: str, passthrough: list[str]) -> list[str]:
    """Arena's arguments: launcher options first, then the scene as an external environment."""
    class_name = "".join(part.capitalize() for part in scene.split("_"))
    return [*passthrough, "--external_environment_class_path", f"i4h_arena.environments:{class_name}", scene]


def _run_diagnostics(scene: str, environment_argv: list[str], run_dir: Path, *, print_json: bool) -> None:
    """Build the environment, run the diagnostics on it and report, all before Isaac Sim closes.

    Closing Isaac Sim ends the process with exit code 0, so a failure is raised
    inside the app context, which Arena turns into exit code 1.
    """
    sys.argv = [sys.argv[0], *environment_argv]
    from isaaclab_arena.cli.isaaclab_arena_cli import get_isaaclab_arena_cli_parser  # noqa: PLC0415
    from isaaclab_arena.utils.isaaclab_utils.simulation_app import SimulationAppContext  # noqa: PLC0415

    parser = get_isaaclab_arena_cli_parser()
    app_args, _ = parser.parse_known_args()
    with SimulationAppContext(app_args):
        from isaaclab_arena_environments.cli import (  # noqa: PLC0415
            get_arena_builder_from_cli,
            get_isaaclab_arena_environments_cli_parser,
        )

        sys.path.insert(0, str(Path(__file__).resolve().parent))
        import robot_pd_diagnostics  # noqa: PLC0415

        parser = get_isaaclab_arena_environments_cli_parser(parser)
        args, hydra_overrides = parser.parse_known_args()
        env = get_arena_builder_from_cli(args, hydra_overrides=hydra_overrides).make_registered()
        env.reset()
        try:
            response = {"ok": True, "result": robot_pd_diagnostics.run(env.unwrapped, scene)}
        except Exception as exc:  # noqa: BLE001 - reported with the run directory before the exit code
            response = {"ok": False, "error": f"{type(exc).__name__}: {exc}"}
        _print_response_summary(response, run_dir)
        if print_json:
            print(json.dumps(response, indent=2), flush=True)
        if not response["ok"]:
            raise RuntimeError("robot PD diagnostics failed")


def main(argv: list[str] | None = None) -> int:
    args, passthrough = _parser().parse_known_args(argv)
    root = Path(__file__).resolve().parents[2]
    timestamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    run_dir = root / "runs" / args.scene / f"{timestamp}_pd_tuning"
    config_path = run_dir / "robot_pd_tuning_config.json"
    config = _generated_config(args)
    environment_argv = _environment_argv(args.scene, passthrough)

    if args.dry_run:
        print(
            json.dumps(
                {
                    "config_path": str(config_path),
                    "run_dir": str(run_dir),
                    "config": config,
                    "arena_args": environment_argv,
                },
                indent=2,
            )
        )
        return 0

    run_dir.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2) + "\n", encoding="utf-8")
    _log(f"[robot-pd] wrote config: {config_path}")
    # robot_pd_tuning_common reads its configuration and writes its results here.
    os.environ["I4H_PD_RUN_DIR"] = str(run_dir)
    _run_diagnostics(args.scene, environment_argv, run_dir, print_json=args.print_json)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))

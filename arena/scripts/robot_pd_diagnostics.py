# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Consolidated live diagnostics for articulated robot joint-drive tuning.

Run this script through the Isaac Sim Python server for any workflow whose
active runner has ``env.scene["robot"]``. Configure modes through the
``robot_pd_tuning_config.json`` in ``I4H_PD_RUN_DIR``:

.. code-block:: json

   {
     "pd_diagnostics": {"modes": ["inspect-usd", "direct-state", "step-response", "trajectory"]},
     "common": {"plots": {"enabled": true}},
     "joint_step_response": {
       "overshoot_target_fraction": 0.01,
       "usd_override": {
         "enabled": true,
         "output_path": "robot_pd_tuned_drive_overlay.usda",
         "solver_position_iterations": 12,
         "solver_velocity_iterations": 4
       }
     }
   }

The individual modes intentionally separate rigging, drive dynamics, and
coordinated joint tracking so PD gain changes are not judged only from an IK
controller trace.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import torch

from i4h_arena.runner import active_runner

_SCRIPT_DIR = Path(globals().get("script_path", __file__)).resolve().parent
if str(_SCRIPT_DIR) not in sys.path:
    sys.path.insert(0, str(_SCRIPT_DIR))

from robot_pd_tuning_common import actuator_summary  # noqa: E402
from robot_pd_tuning_common import (  # noqa: E402
    apply_gain_overrides,
    clamp_target_to_soft_limits,
    command_joint_targets_once,
    compute_step_metrics,
    hold_joint_targets,
    joint_offset_value,
    joint_unit,
    load_config,
    plot_tracked_csv_by_joint,
    recommend_pd,
    reset_to_default_joint_state,
    result_paths,
    robot_context,
    robot_metadata_summary,
    run_dir,
    write_csv,
    write_json,
    write_usd_drive_overlay,
)

_RUNNER = active_runner()
_RUNNER_ENV = _RUNNER.env
_WORKFLOW_NAME = _RUNNER.workflow.name

_JOINT_DRIVE_MODES = ("inspect-usd", "direct-state", "step-response", "trajectory")
_MODE_ALIASES = {
    "inspect": "inspect-usd",
    "inspect-usd": "inspect-usd",
    "usd": "inspect-usd",
    "usd-inspect": "inspect-usd",
    "usd_inspect": "inspect-usd",
    "direct": "direct-state",
    "direct_state": "direct-state",
    "direct-state": "direct-state",
    "step": "step-response",
    "step_response": "step-response",
    "step-response": "step-response",
    "joint-step": "step-response",
    "trajectory": "trajectory",
    "joint-trajectory": "trajectory",
}

_DIAGNOSTIC_DEFAULTS = {"modes": ["all"]}
_DIRECT_DEFAULTS = {
    "joint_offsets": None,
    "joint_offsets_rad": None,
    "default_revolute_joint_offset_rad": 0.10,
    "gain_overrides": {},
}
_STEP_DEFAULTS = {
    "step_amplitude_rad": 0.10,
    "step_amplitude_m": 0.02,
    "step_amplitudes": None,
    "warmup_s": 0.25,
    "duration_s": 2.0,
    "position_tolerance_rad": 0.01,
    "velocity_tolerance_rad_s": 0.03,
    "target_rise_time_s": 0.45,
    "overshoot_target_fraction": 0.01,
    "sample_every": 1,
    "gain_overrides": {},
}
_TRAJECTORY_DEFAULTS = {
    "joint_offsets": None,
    "joint_offsets_rad": None,
    "default_revolute_joint_offset_rad": 0.16,
    "command_velocity_rad_s": 2.5,
    "warmup_s": 0.25,
    "hold_s": 0.75,
    "sample_every": 1,
    "gain_overrides": {},
}
_PLOT_DEFAULTS = {
    "enabled": False,
    "output_dir": None,
    "format": "png",
    "group_column": "joint",
    "x_column": "time_s",
}
_USD_OVERRIDE_DEFAULTS = {
    "enabled": False,
    "output_path": None,
    "overwrite": True,
    "include_joint_friction": True,
    "include_max_joint_velocity": True,
    "solver_position_iterations": None,
    "solver_velocity_iterations": None,
    "articulation_prim_paths": None,
}


@torch.no_grad()
def _run() -> dict[str, Any]:
    modes = _selected_modes()
    reports = {}
    for mode in modes:
        if mode == "inspect-usd":
            reports[mode] = _run_inspect_usd()
        elif mode == "direct-state":
            reports[mode] = _run_direct_state()
        elif mode == "step-response":
            reports[mode] = _run_step_response()
        elif mode == "trajectory":
            reports[mode] = _run_trajectory_response()
        else:
            raise ValueError(f"unsupported robot PD diagnostic mode: {mode}")

    json_path, _ = result_paths("robot_pd_diagnostics")
    report = {
        "test": "robot_pd_diagnostics",
        "env_id": _WORKFLOW_NAME,
        "modes": modes,
        "results": reports,
        "outputs": {
            "json": str(json_path),
            "mode_outputs": {mode: reports[mode].get("outputs", {}) for mode in modes},
        },
    }
    write_json(json_path, report)
    return report


def _selected_modes() -> list[str]:
    cfg = load_config("pd_diagnostics", _DIAGNOSTIC_DEFAULTS)
    requested = cfg.get("modes", ["all"])
    if isinstance(requested, str):
        requested = [requested]
    if not requested:
        raise ValueError("pd_diagnostics.modes must contain at least one mode")

    modes: list[str] = []
    for raw_mode in requested:
        mode = str(raw_mode).strip().lower()
        if mode == "all":
            for core_mode in _JOINT_DRIVE_MODES:
                if core_mode not in modes:
                    modes.append(core_mode)
            continue
        canonical = _MODE_ALIASES.get(mode)
        if canonical is None:
            raise ValueError(
                f"unknown robot PD diagnostic mode {raw_mode!r}; expected one of {_JOINT_DRIVE_MODES} or 'all'"
            )
        if canonical not in modes:
            modes.append(canonical)
    return modes


def _run_inspect_usd() -> dict[str, Any]:
    cfg = load_config("inspect_usd", {})
    _, robot, joint_names, joint_ids = robot_context(_RUNNER_ENV, cfg)
    gains = actuator_summary(robot, joint_names, joint_ids)
    json_path, _ = result_paths("robot_usd_joint_drive_probe")
    report = {
        "test": "robot_usd_joint_drive_probe",
        "env_id": _WORKFLOW_NAME,
        "joint_order": joint_names,
        "actuators": gains,
        "robot_usd": robot_metadata_summary(robot, joint_names),
        "outputs": {"json": str(json_path)},
    }
    write_json(json_path, report)
    return report


def _run_direct_state() -> dict[str, Any]:
    cfg = load_config("direct_joint_state_probe", _DIRECT_DEFAULTS)
    u, robot, joint_names, joint_ids = robot_context(_RUNNER_ENV, cfg)
    apply_gain_overrides(robot, joint_names, joint_ids, cfg.get("gain_overrides"))
    gains = actuator_summary(robot, joint_names, joint_ids)
    csv_rows = []
    results = {}
    for case_name, target in _direct_state_cases(cfg, u, robot, joint_names, joint_ids):
        zeros = torch.zeros_like(target)
        robot.write_joint_position_to_sim_index(position=target, joint_ids=joint_ids)
        robot.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=joint_ids)
        robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)
        robot.set_joint_velocity_target_index(target=zeros, joint_ids=joint_ids)
        u.scene.write_data_to_sim()
        u.sim.step(render=False)
        u.scene.update(dt=u.physics_dt)
        actual = robot.data.joint_pos[0, joint_ids].detach()
        desired = target[0].detach()
        error = desired - actual
        results[case_name] = {
            "max_abs_joint_error_rad": float(torch.max(torch.abs(error)).item()),
            "joint_targets_rad": {name: float(desired[i].item()) for i, name in enumerate(joint_names)},
            "joint_actual_rad": {name: float(actual[i].item()) for i, name in enumerate(joint_names)},
            "tracked_frames": _tracked_frame_poses(u, cfg),
        }
        for local_id, name in enumerate(joint_names):
            csv_rows.append(
                {
                    "case": case_name,
                    "joint": name,
                    "target_rad": float(desired[local_id].item()),
                    "actual_rad": float(actual[local_id].item()),
                    "error_rad": float(error[local_id].item()),
                }
            )
    json_path, csv_path = result_paths("robot_direct_joint_state_probe")
    write_csv(csv_path, csv_rows, ["case", "joint", "target_rad", "actual_rad", "error_rad"])
    outputs = {"json": str(json_path), "csv": str(csv_path)}
    plots = _plot_csv_if_requested(csv_path, cfg, default_output_dir_name="robot_direct_joint_state_probe_plots")
    if plots:
        outputs["plots"] = plots
    report = {
        "test": "direct_joint_state_probe",
        "env_id": _WORKFLOW_NAME,
        "config": cfg,
        "joint_order": joint_names,
        "actuators": gains,
        "robot_usd": robot_metadata_summary(robot, joint_names),
        "results": results,
        "outputs": outputs,
    }
    write_json(json_path, report)
    return report


def _direct_state_cases(cfg: dict[str, Any], u, robot, joint_names: list[str], joint_ids: list[int]):
    offsets = cfg.get("joint_offsets")
    if offsets is None:
        offsets = cfg.get("joint_offsets_rad")
    for local_id, name in enumerate(joint_names):
        base = reset_to_default_joint_state(u, robot, joint_ids)
        target = base.clone()
        target[:, local_id] += joint_offset_value(
            robot,
            name,
            local_id,
            offsets,
            float(cfg["default_revolute_joint_offset_rad"]),
        )
        yield name, clamp_target_to_soft_limits(robot, joint_ids, target)
    base = reset_to_default_joint_state(u, robot, joint_ids)
    target = base.clone()
    for local_id, name in enumerate(joint_names):
        target[:, local_id] += joint_offset_value(
            robot,
            name,
            local_id,
            offsets,
            float(cfg["default_revolute_joint_offset_rad"]),
        )
    yield "all_joints", clamp_target_to_soft_limits(robot, joint_ids, target)


def _run_step_response() -> dict[str, Any]:
    cfg = load_config("joint_step_response", _STEP_DEFAULTS)
    u, robot, joint_names, joint_ids = robot_context(_RUNNER_ENV, cfg)
    apply_gain_overrides(robot, joint_names, joint_ids, cfg.get("gain_overrides"))
    gains = actuator_summary(robot, joint_names, joint_ids)
    sim_dt = float(u.physics_dt)
    warmup_steps = round(float(cfg["warmup_s"]) / sim_dt)
    duration_steps = round(float(cfg["duration_s"]) / sim_dt)
    sample_every = max(1, int(cfg["sample_every"]))
    csv_rows = []
    joint_results = {}
    for local_id, name in enumerate(joint_names):
        base = reset_to_default_joint_state(u, robot, joint_ids)
        hold_joint_targets(u, robot, joint_ids, base, warmup_steps)
        target = _step_target(robot, joint_ids, base, local_id, _step_amplitude(robot, name, local_id, cfg))
        start = float(base[0, local_id].item())
        target_value = float(target[0, local_id].item())
        times: list[float] = []
        positions: list[float] = []
        velocities: list[float] = []
        for step in range(duration_steps):
            command_joint_targets_once(u, robot, joint_ids, target)
            if step % sample_every == 0 or step == duration_steps - 1:
                t = (step + 1) * sim_dt
                pos = float(robot.data.joint_pos[0, joint_ids[local_id]].item())
                vel = float(robot.data.joint_vel[0, joint_ids[local_id]].item())
                err = target_value - pos
                times.append(t)
                positions.append(pos)
                velocities.append(vel)
                csv_rows.append(
                    {
                        "joint": name,
                        "time_s": t,
                        "target_rad": target_value,
                        "position_rad": pos,
                        "velocity_rad_s": vel,
                        "error_rad": err,
                    }
                )
        velocity_limit = gains[name].get("velocity_limit_sim") or gains[name].get("soft_velocity_limit_rad_s")
        metrics = compute_step_metrics(
            times,
            positions,
            velocities,
            start,
            target_value,
            position_tolerance_rad=float(cfg["position_tolerance_rad"]),
            velocity_tolerance_rad_s=float(cfg["velocity_tolerance_rad_s"]),
            velocity_limit_rad_s=velocity_limit,
        )
        overshoot_target = float(cfg["overshoot_target_fraction"])
        metrics["overshoot_target_fraction"] = overshoot_target
        metrics["overshoot_passed"] = float(metrics["overshoot_pct"]) <= overshoot_target
        metrics["position_tolerance_rad"] = float(cfg["position_tolerance_rad"])
        metrics["velocity_tolerance_rad_s"] = float(cfg["velocity_tolerance_rad_s"])
        joint_results[name] = {
            "metrics": metrics,
            "gains": gains[name],
            "recommendation": recommend_pd(
                name,
                metrics,
                gains[name],
                target_rise_time_s=float(cfg["target_rise_time_s"]),
                position_tolerance_rad=float(cfg["position_tolerance_rad"]),
                overshoot_target_fraction=overshoot_target,
            ),
        }
    json_path, csv_path = result_paths("robot_joint_step_response")
    write_csv(csv_path, csv_rows, ["joint", "time_s", "target_rad", "position_rad", "velocity_rad_s", "error_rad"])
    outputs = {"json": str(json_path), "csv": str(csv_path)}
    plots = _plot_csv_if_requested(
        csv_path,
        cfg,
        default_output_dir_name="robot_joint_step_response_plots",
        overshoot_by_group={name: float(result["metrics"]["overshoot_pct"]) for name, result in joint_results.items()},
        rise_time_by_group={name: result["metrics"]["rise_time_90_s"] for name, result in joint_results.items()},
        metrics_by_group={name: result["metrics"] for name, result in joint_results.items()},
    )
    if plots:
        outputs["plots"] = plots
    usd_overlay = _write_usd_override_if_requested(robot, joint_names, joint_results, cfg)
    if usd_overlay:
        outputs["usd_overlay"] = usd_overlay
    report = {
        "test": "joint_step_response",
        "env_id": _WORKFLOW_NAME,
        "config": cfg,
        "sim_dt": sim_dt,
        "joint_order": joint_names,
        "actuators": gains,
        "robot_usd": robot_metadata_summary(robot, joint_names),
        "results": joint_results,
        "outputs": outputs,
    }
    write_json(json_path, report)
    return report


def _step_target(
    robot, joint_ids: list[int], base: torch.Tensor, local_joint_id: int, amplitude: float
) -> torch.Tensor:
    target = base.clone()
    target[:, local_joint_id] += amplitude
    target = clamp_target_to_soft_limits(robot, joint_ids, target)
    actual = float((target[0, local_joint_id] - base[0, local_joint_id]).item())
    if abs(actual) < 0.5 * abs(amplitude):
        target = base.clone()
        target[:, local_joint_id] -= amplitude
        target = clamp_target_to_soft_limits(robot, joint_ids, target)
    return target


def _step_amplitude(robot, joint_name: str, local_id: int, cfg: dict[str, Any]) -> float:
    amplitudes = cfg.get("step_amplitudes")
    if amplitudes is not None:
        return joint_offset_value(robot, joint_name, local_id, amplitudes, float(cfg["step_amplitude_rad"]))
    if joint_unit(robot, joint_name) == "m":
        return 0.02 if cfg.get("step_amplitude_m") is None else float(cfg["step_amplitude_m"])
    return joint_offset_value(
        robot,
        joint_name,
        local_id,
        None,
        float(cfg["step_amplitude_rad"]),
    )


def _run_trajectory_response() -> dict[str, Any]:
    cfg = load_config("joint_trajectory_response", _TRAJECTORY_DEFAULTS)
    u, robot, joint_names, joint_ids = robot_context(_RUNNER_ENV, cfg)
    apply_gain_overrides(robot, joint_names, joint_ids, cfg.get("gain_overrides"))
    gains = actuator_summary(robot, joint_names, joint_ids)
    sim_dt = float(u.physics_dt)
    sample_every = max(1, int(cfg["sample_every"]))
    base = reset_to_default_joint_state(u, robot, joint_ids)
    hold_joint_targets(u, robot, joint_ids, base, round(float(cfg["warmup_s"]) / sim_dt))
    offsets = torch.zeros_like(base)
    offset_cfg = cfg.get("joint_offsets")
    if offset_cfg is None:
        offset_cfg = cfg.get("joint_offsets_rad")
    for local_id, name in enumerate(joint_names):
        offsets[:, local_id] = joint_offset_value(
            robot,
            name,
            local_id,
            offset_cfg,
            float(cfg["default_revolute_joint_offset_rad"]),
        )
    goal = clamp_target_to_soft_limits(robot, joint_ids, base + offsets)
    command = base.clone()
    max_delta = float(cfg["command_velocity_rad_s"]) * sim_dt
    hold_steps = round(float(cfg["hold_s"]) / sim_dt)
    csv_rows = []
    summaries = {name: {"max_abs_error_rad": 0.0, "max_abs_velocity_rad_s": 0.0} for name in joint_names}
    step = 0
    while True:
        delta = goal - command
        if torch.max(torch.abs(delta)).item() <= 1.0e-6:
            break
        command = command + torch.clamp(delta, min=-max_delta, max=max_delta)
        command_joint_targets_once(u, robot, joint_ids, command)
        if step % sample_every == 0:
            _append_joint_rows(csv_rows, summaries, robot, joint_names, joint_ids, command, step * sim_dt, "ramp")
        step += 1
    for hold_step in range(hold_steps):
        command_joint_targets_once(u, robot, joint_ids, goal)
        if hold_step % sample_every == 0 or hold_step == hold_steps - 1:
            _append_joint_rows(
                csv_rows, summaries, robot, joint_names, joint_ids, goal, (step + hold_step) * sim_dt, "hold"
            )
    final_pos = robot.data.joint_pos[0, joint_ids].detach()
    final_vel = robot.data.joint_vel[0, joint_ids].detach()
    for local_id, name in enumerate(joint_names):
        summaries[name].update(
            {
                "start_rad": float(base[0, local_id].item()),
                "goal_rad": float(goal[0, local_id].item()),
                "final_rad": float(final_pos[local_id].item()),
                "final_error_rad": float((goal[0, local_id] - final_pos[local_id]).item()),
                "final_velocity_rad_s": float(final_vel[local_id].item()),
                "velocity_limit_sim": gains[name].get("velocity_limit_sim"),
                "velocity_limit_fraction": (
                    summaries[name]["max_abs_velocity_rad_s"] / gains[name]["velocity_limit_sim"]
                    if gains[name].get("velocity_limit_sim")
                    else None
                ),
            }
        )
    json_path, csv_path = result_paths("robot_joint_trajectory_response")
    write_csv(
        csv_path,
        csv_rows,
        ["phase", "time_s", "joint", "target_rad", "position_rad", "velocity_rad_s", "error_rad"],
    )
    outputs = {"json": str(json_path), "csv": str(csv_path)}
    plots = _plot_csv_if_requested(csv_path, cfg, default_output_dir_name="robot_joint_trajectory_response_plots")
    if plots:
        outputs["plots"] = plots
    report = {
        "test": "joint_trajectory_response",
        "env_id": _WORKFLOW_NAME,
        "config": cfg,
        "sim_dt": sim_dt,
        "joint_order": joint_names,
        "actuators": gains,
        "robot_usd": robot_metadata_summary(robot, joint_names),
        "results": summaries,
        "outputs": outputs,
    }
    write_json(json_path, report)
    return report


def _plot_csv_if_requested(
    csv_path: Path,
    cfg: dict[str, Any],
    *,
    default_output_dir_name: str,
    overshoot_by_group: dict[str, float] | None = None,
    rise_time_by_group: dict[str, float | None] | None = None,
    metrics_by_group: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    plot_cfg = dict(_PLOT_DEFAULTS)
    if isinstance(cfg.get("plots"), dict):
        plot_cfg.update(cfg["plots"])
    if not plot_cfg.get("enabled"):
        return {}
    output_dir = plot_cfg.get("output_dir")
    output_dir = _resolve_run_output_path(output_dir) if output_dir else run_dir() / default_output_dir_name
    return plot_tracked_csv_by_joint(
        csv_path,
        output_dir=Path(output_dir),
        group_column=str(plot_cfg.get("group_column") or "joint"),
        x_column=str(plot_cfg.get("x_column") or "time_s"),
        image_format=str(plot_cfg.get("format") or "png"),
        overshoot_by_group=overshoot_by_group,
        rise_time_by_group=rise_time_by_group,
        metrics_by_group=metrics_by_group,
    )


def _write_usd_override_if_requested(
    robot,
    joint_names: list[str],
    joint_results: dict[str, dict[str, Any]],
    cfg: dict[str, Any],
) -> dict[str, Any]:
    overlay_cfg = dict(_USD_OVERRIDE_DEFAULTS)
    if isinstance(cfg.get("usd_override"), dict):
        overlay_cfg.update(cfg["usd_override"])
    if not overlay_cfg.get("enabled"):
        return {}
    output_path = _resolve_run_output_path(overlay_cfg.get("output_path") or "robot_pd_tuned_drive_overlay.usda")
    joint_gains: dict[str, dict[str, float | None]] = {}
    for joint_name in joint_names:
        result = joint_results[joint_name]
        recommendation = result["recommendation"]
        gains = result["gains"]
        joint_gains[joint_name] = {
            "stiffness": float(recommendation["recommended_stiffness"]),
            "damping": float(recommendation["recommended_damping"]),
        }
        if overlay_cfg.get("include_joint_friction") and gains.get("friction") is not None:
            joint_gains[joint_name]["joint_friction"] = float(gains["friction"])
        velocity_limit = _positive_float(gains.get("velocity_limit_sim"))
        if velocity_limit is None:
            velocity_limit = _positive_float(gains.get("soft_velocity_limit_rad_s"))
        if overlay_cfg.get("include_max_joint_velocity") and velocity_limit is not None:
            joint_gains[joint_name]["max_joint_velocity_sim"] = velocity_limit
    return write_usd_drive_overlay(
        robot,
        joint_gains,
        Path(output_path),
        solver_position_iterations=_optional_int(overlay_cfg.get("solver_position_iterations")),
        solver_velocity_iterations=_optional_int(overlay_cfg.get("solver_velocity_iterations")),
        articulation_prim_paths=_optional_str_list(overlay_cfg.get("articulation_prim_paths")),
        include_joint_friction=bool(overlay_cfg.get("include_joint_friction")),
        overwrite=bool(overlay_cfg.get("overwrite")),
    )


def _resolve_run_output_path(value: Any) -> Path:
    path = Path(str(value)).expanduser()
    if not path.is_absolute():
        path = run_dir() / path
    return path


def _append_joint_rows(
    rows: list[dict[str, Any]],
    summaries: dict[str, dict[str, float]],
    robot,
    joint_names: list[str],
    joint_ids: list[int],
    command: torch.Tensor,
    time_s: float,
    phase: str,
) -> None:
    pos = robot.data.joint_pos[0, joint_ids].detach()
    vel = robot.data.joint_vel[0, joint_ids].detach()
    for local_id, name in enumerate(joint_names):
        err = float((command[0, local_id] - pos[local_id]).item())
        speed = abs(float(vel[local_id].item()))
        summaries[name]["max_abs_error_rad"] = max(summaries[name]["max_abs_error_rad"], abs(err))
        summaries[name]["max_abs_velocity_rad_s"] = max(summaries[name]["max_abs_velocity_rad_s"], speed)
        rows.append(
            {
                "phase": phase,
                "time_s": float(time_s),
                "joint": name,
                "target_rad": float(command[0, local_id].item()),
                "position_rad": float(pos[local_id].item()),
                "velocity_rad_s": float(vel[local_id].item()),
                "error_rad": err,
            }
        )


def _tracked_frame_poses(u, cfg: dict[str, Any]) -> dict[str, dict[str, list[float]]]:
    frame_names = cfg.get("tracked_frames", [])
    if isinstance(frame_names, str):
        frame_names = [frame_names]
    out = {}
    for frame_name in frame_names:
        try:
            frame = u.scene[str(frame_name)].data
        except Exception:  # noqa: BLE001, S112 - optional scene frames may not exist
            continue
        out[str(frame_name)] = {
            "pos_w": _float_list(frame.target_pos_w[0, 0, :]),
            "quat_wxyz": _float_list(frame.target_quat_w[0, 0, :]),
        }
    return out


def _positive_float(value: Any) -> float | None:
    if value is None:
        return None
    out = float(value)
    if not math.isfinite(out) or out <= 0.0:
        return None
    return out


def _optional_int(value: Any) -> int | None:
    return None if value is None else int(value)


def _optional_str_list(value: Any) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, str):
        return [value]
    return [str(item) for item in value]


def _float_list(tensor: torch.Tensor) -> list[float]:
    return [float(v) for v in tensor.detach().cpu().flatten().tolist()]


result = _run()
print(json.dumps(result, indent=2))

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Shared live-simulator helpers for robot joint-drive tuning diagnostics."""

from __future__ import annotations

import csv
import json
import logging
import math
import os
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import torch

_REPO_ROOT = Path(__file__).resolve().parents[2]
_RUNS_ROOT = _REPO_ROOT / "runs"
_DEFAULT_RUN_DIR = Path(os.environ.get("I4H_PD_RUN_DIR", _RUNS_ROOT / ".latest"))
_CONFIG_PATH = _DEFAULT_RUN_DIR / "robot_pd_tuning_config.json"
_USD_METADATA_CACHE: dict[str, RobotUsdMetadata] = {}


@dataclass(frozen=True)
class JointDriveMetadata:
    name: str
    path: str
    joint_type: str
    drive_axis: str | None
    unit: str
    stiffness: float | None
    damping: float | None
    max_force: float | None
    max_joint_velocity: float | None
    joint_friction: float | None
    body0: list[str]
    body1: list[str]


@dataclass(frozen=True)
class RobotUsdMetadata:
    usd_path: str | None
    default_prim: str | None
    articulation_roots: list[str]
    joints: dict[str, JointDriveMetadata]
    load_error: str | None = None


def run_dir() -> Path:
    _DEFAULT_RUN_DIR.mkdir(parents=True, exist_ok=True)
    return _DEFAULT_RUN_DIR.resolve()


def load_config(section: str, defaults: dict[str, Any]) -> dict[str, Any]:
    config = dict(defaults)
    if _CONFIG_PATH.exists():
        payload = json.loads(_CONFIG_PATH.read_text(encoding="utf-8"))
        config.update(payload.get("common", {}))
        config.update(payload.get(section, {}))
    return config


def unwrap_env(env: Any) -> Any:
    return getattr(env, "unwrapped", env)


def robot_context(env: Any, cfg: dict[str, Any] | None = None) -> tuple[Any, Any, list[str], list[int]]:
    u = unwrap_env(env)
    robot = u.scene["robot"]
    joint_names = discover_joint_names(robot, cfg or {})
    joint_ids = [robot.data.joint_names.index(name) for name in joint_names]
    return u, robot, joint_names, joint_ids


def robot_usd_path(robot: Any) -> str | None:
    return maybe_str(getattr(getattr(robot.cfg, "spawn", None), "usd_path", None))


def maybe_str(value: Any) -> str | None:
    if value is None:
        return None
    try:
        text = str(value)
    except (TypeError, ValueError):
        return None
    return text or None


def load_robot_usd_metadata(robot: Any) -> RobotUsdMetadata:
    usd_path = robot_usd_path(robot)
    if not usd_path:
        return RobotUsdMetadata(None, None, [], {}, "robot.cfg.spawn.usd_path is not set")
    if usd_path in _USD_METADATA_CACHE:
        return _USD_METADATA_CACHE[usd_path]
    try:
        from pxr import PhysxSchema, Usd, UsdPhysics
    except ModuleNotFoundError as exc:
        return RobotUsdMetadata(usd_path, None, [], {}, f"pxr modules unavailable: {exc}")

    try:
        stage = Usd.Stage.Open(usd_path)
    except Exception as exc:  # noqa: BLE001 - pxr may raise multiple exception types
        return RobotUsdMetadata(usd_path, None, [], {}, f"could not open robot USD: {exc}")
    if stage is None:
        return RobotUsdMetadata(usd_path, None, [], {}, "Usd.Stage.Open returned None")

    default_prim = stage.GetDefaultPrim()
    default_path = str(default_prim.GetPath()) if default_prim and default_prim.IsValid() else None
    articulation_roots = [
        str(prim.GetPath()) for prim in stage.Traverse() if prim.HasAPI(UsdPhysics.ArticulationRootAPI)
    ]
    joints: dict[str, JointDriveMetadata] = {}
    for prim in stage.Traverse():
        if not prim.IsA(UsdPhysics.Joint):
            continue
        metadata = _joint_metadata(prim, UsdPhysics, PhysxSchema)
        if metadata is not None:
            joints[metadata.name] = metadata
    metadata = RobotUsdMetadata(usd_path, default_path, articulation_roots, joints)
    _USD_METADATA_CACHE[usd_path] = metadata
    return metadata


def _joint_metadata(prim: Any, UsdPhysics: Any, PhysxSchema: Any) -> JointDriveMetadata | None:
    joint_type = prim.GetTypeName()
    drive_axis = _authored_drive_axis(prim, UsdPhysics)
    if drive_axis is None:
        return None
    drive = UsdPhysics.DriveAPI.Get(prim, drive_axis)
    joint = UsdPhysics.Joint(prim)
    physx_joint = PhysxSchema.PhysxJointAPI(prim)
    unit = "m" if "Prismatic" in joint_type else "rad"
    return JointDriveMetadata(
        name=prim.GetName(),
        path=str(prim.GetPath()),
        joint_type=joint_type,
        drive_axis=drive_axis,
        unit=unit,
        stiffness=maybe_float(drive.GetStiffnessAttr().Get()),
        damping=maybe_float(drive.GetDampingAttr().Get()),
        max_force=maybe_float(drive.GetMaxForceAttr().Get()),
        max_joint_velocity=maybe_float(physx_joint.GetMaxJointVelocityAttr().Get()),
        joint_friction=maybe_float(physx_joint.GetJointFrictionAttr().Get()),
        body0=[str(path) for path in joint.GetBody0Rel().GetTargets()],
        body1=[str(path) for path in joint.GetBody1Rel().GetTargets()],
    )


def _authored_drive_axis(prim: Any, UsdPhysics: Any) -> str | None:
    property_names = set(prim.GetPropertyNames())
    for axis in ("angular", "linear", "rotX", "rotY", "rotZ", "transX", "transY", "transZ"):
        prefix = f"drive:{axis}:physics:"
        if any(name.startswith(prefix) for name in property_names):
            return axis
        drive = UsdPhysics.DriveAPI.Get(prim, axis)
        attrs = (drive.GetStiffnessAttr(), drive.GetDampingAttr(), drive.GetMaxForceAttr(), drive.GetTypeAttr())
        if any(attr and attr.HasAuthoredValueOpinion() for attr in attrs):
            return axis
    return None


def discover_joint_names(robot: Any, cfg: dict[str, Any]) -> list[str]:
    all_joint_names = list(robot.data.joint_names)
    requested = cfg.get("joint_names")
    if requested:
        if isinstance(requested, str):
            requested = [requested]
        return _validate_joint_names(all_joint_names, [str(name) for name in requested])

    exprs = cfg.get("joint_names_expr") or cfg.get("joint_name_expr")
    if exprs:
        if isinstance(exprs, str):
            exprs = [exprs]
        selected = [name for name in all_joint_names if any(_expr_matches(expr, name) for expr in exprs)]
        return _validate_joint_names(all_joint_names, selected)

    metadata = load_robot_usd_metadata(robot)
    usd_joint_names = set(metadata.joints)
    if usd_joint_names:
        selected = [name for name in all_joint_names if name in usd_joint_names]
        if selected:
            return selected
    return all_joint_names


def _validate_joint_names(all_joint_names: list[str], selected: list[str]) -> list[str]:
    missing = [name for name in selected if name not in all_joint_names]
    if missing:
        raise ValueError(
            f"requested joints are not present in loaded articulation: {missing}; available={all_joint_names}"
        )
    if not selected:
        raise ValueError(f"joint selection matched no joints; available={all_joint_names}")
    return selected


def robot_metadata_summary(robot: Any, joint_names: list[str]) -> dict[str, Any]:
    metadata = load_robot_usd_metadata(robot)
    payload = asdict(metadata)
    payload["selected_joints"] = {
        name: asdict(metadata.joints[name]) if name in metadata.joints else None for name in joint_names
    }
    return payload


def joint_unit(robot: Any, joint_name: str) -> str:
    metadata = load_robot_usd_metadata(robot)
    joint = metadata.joints.get(joint_name)
    return "rad" if joint is None else joint.unit


def runtime_velocity_limit_to_usd(robot: Any, joint_name: str, value: float) -> float:
    """Convert a runtime joint velocity limit to the unit expected by USD PhysX."""

    if joint_unit(robot, joint_name) == "rad":
        return math.degrees(value)
    return value


def joint_offset_value(robot: Any, joint_name: str, local_id: int, offsets: Any, default_revolute: float) -> float:
    if isinstance(offsets, dict):
        if joint_name in offsets:
            value = float(offsets[joint_name])
        elif "default" in offsets:
            value = float(offsets["default"])
        else:
            value = _default_joint_offset(robot, joint_name, local_id, default_revolute)
    elif offsets is None:
        value = _default_joint_offset(robot, joint_name, local_id, default_revolute)
    else:
        value = float(offsets)
    return value


def _default_joint_offset(robot: Any, joint_name: str, local_id: int, default_revolute: float) -> float:
    magnitude = 0.02 if joint_unit(robot, joint_name) == "m" else default_revolute
    return magnitude if local_id % 2 == 0 else -magnitude


def maybe_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def actuator_summary(robot: Any, joint_names: list[str], joint_ids: list[int]) -> dict[str, dict[str, float | None]]:
    out: dict[str, dict[str, float | None]] = {}
    stiffness = robot.data.joint_stiffness[0, joint_ids]
    damping = robot.data.joint_damping[0, joint_ids]
    armature = robot.data.joint_armature[0, joint_ids]
    friction = robot.data.joint_friction_coeff[0, joint_ids]
    velocity_limits = robot.data.soft_joint_vel_limits[0, joint_ids]
    efforts_by_joint: dict[str, float | None] = {name: None for name in joint_names}
    velocity_limit_sim_by_joint: dict[str, float | None] = {name: None for name in joint_names}
    for actuator in robot.cfg.actuators.values():
        for expr in actuator.joint_names_expr:
            for joint_name in joint_names:
                if _expr_matches(expr, joint_name):
                    effort = _actuator_value_for_joint(actuator, "effort_limit_sim", joint_name)
                    if effort is None:
                        effort = _actuator_value_for_joint(actuator, "effort_limit", joint_name)
                    efforts_by_joint[joint_name] = effort
                    velocity_limit_sim_by_joint[joint_name] = _actuator_value_for_joint(
                        actuator, "velocity_limit_sim", joint_name
                    )
    for local_id, name in enumerate(joint_names):
        out[name] = {
            "stiffness": float(stiffness[local_id].item()),
            "damping": float(damping[local_id].item()),
            "armature": float(armature[local_id].item()),
            "friction": float(friction[local_id].item()),
            "effort_limit": efforts_by_joint[name],
            "soft_velocity_limit_rad_s": float(velocity_limits[local_id].item()),
            "velocity_limit_sim": velocity_limit_sim_by_joint[name],
        }
    return out


def apply_gain_overrides(
    robot: Any, joint_names: list[str], joint_ids: list[int], gains: dict[str, Any] | None
) -> None:
    if not gains:
        return
    stiffness = robot.data.joint_stiffness[:, joint_ids].clone()
    damping = robot.data.joint_damping[:, joint_ids].clone()
    friction = robot.data.joint_friction_coeff[:, joint_ids].clone()
    has_friction = False
    for local_id, name in enumerate(joint_names):
        entry = _gain_entry(gains, name)
        if "stiffness" in entry:
            stiffness[:, local_id] = float(entry["stiffness"])
        if "damping" in entry:
            damping[:, local_id] = float(entry["damping"])
        if "friction" in entry or "joint_friction" in entry:
            friction[:, local_id] = float(entry.get("friction", entry.get("joint_friction")))
            has_friction = True
    robot.write_joint_stiffness_to_sim_index(stiffness=stiffness, joint_ids=joint_ids)
    robot.write_joint_damping_to_sim_index(damping=damping, joint_ids=joint_ids)
    if has_friction:
        robot.write_joint_friction_coefficient_to_sim_index(
            joint_friction_coeff=friction,
            joint_ids=joint_ids,
        )


def _gain_entry(gains: dict[str, Any], joint_name: str) -> dict[str, Any]:
    for key in (joint_name, "default", "*"):
        entry = gains.get(key)
        if isinstance(entry, dict):
            return entry
    for expr, entry in gains.items():
        if isinstance(entry, dict) and _expr_matches(str(expr), joint_name):
            return entry
    return {}


def _actuator_value_for_joint(actuator: Any, field: str, joint_name: str) -> float | None:
    value = getattr(actuator, field, None)
    if isinstance(value, dict):
        if joint_name in value:
            return maybe_float(value[joint_name])
        for expr, expr_value in value.items():
            if _expr_matches(str(expr), joint_name):
                return maybe_float(expr_value)
        return None
    return maybe_float(value)


def _expr_matches(expr: Any, joint_name: str) -> bool:
    expr_text = str(expr)
    if expr_text == joint_name or expr_text == "*":
        return True
    try:
        return re.fullmatch(expr_text, joint_name) is not None
    except re.error:
        pattern = "^" + re.escape(expr_text).replace("\\*", ".*") + "$"
        return re.fullmatch(pattern, joint_name) is not None


def reset_to_default_joint_state(u: Any, robot: Any, joint_ids: list[int]) -> torch.Tensor:
    u.reset()
    target = robot.data.default_joint_pos[:, joint_ids].clone()
    zeros = torch.zeros_like(target)
    robot.write_joint_position_to_sim_index(position=target, joint_ids=joint_ids)
    robot.write_joint_velocity_to_sim_index(velocity=zeros, joint_ids=joint_ids)
    robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)
    robot.set_joint_velocity_target_index(target=zeros, joint_ids=joint_ids)
    u.scene.write_data_to_sim()
    u.sim.step(render=False)
    u.scene.update(dt=u.physics_dt)
    return target


def low_level_step(u: Any) -> None:
    u.scene.write_data_to_sim()
    u.sim.step(render=False)
    u.scene.update(dt=u.physics_dt)


def hold_joint_targets(u: Any, robot: Any, joint_ids: list[int], target: torch.Tensor, steps: int) -> None:
    zeros = torch.zeros_like(target)
    for _ in range(max(0, int(steps))):
        robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)
        robot.set_joint_velocity_target_index(target=zeros, joint_ids=joint_ids)
        low_level_step(u)


def command_joint_targets_once(u: Any, robot: Any, joint_ids: list[int], target: torch.Tensor) -> None:
    zeros = torch.zeros_like(target)
    robot.set_joint_position_target_index(target=target, joint_ids=joint_ids)
    robot.set_joint_velocity_target_index(target=zeros, joint_ids=joint_ids)
    low_level_step(u)


def clamp_target_to_soft_limits(robot: Any, joint_ids: list[int], target: torch.Tensor) -> torch.Tensor:
    limits = robot.data.soft_joint_pos_limits[:, joint_ids, :]
    return torch.maximum(torch.minimum(target, limits[..., 1]), limits[..., 0])


def compute_step_metrics(
    times: list[float],
    positions: list[float],
    velocities: list[float],
    start: float,
    target: float,
    *,
    position_tolerance_rad: float,
    velocity_tolerance_rad_s: float,
    velocity_limit_rad_s: float | None,
) -> dict[str, Any]:
    delta = target - start
    errors = [target - p for p in positions]
    abs_delta = abs(delta)
    normalized = [0.0 for _ in positions] if abs_delta < 1e-9 else [(p - start) / delta for p in positions]
    peak_norm = max(normalized) if normalized else 0.0
    overshoot_rad = max(0.0, peak_norm - 1.0) * abs_delta
    overshoot_pct = 0.0 if abs_delta < 1e-9 else overshoot_rad / abs_delta
    rise_time_s = None
    for t, y in zip(times, normalized, strict=False):
        if y >= 0.9:
            rise_time_s = t
            break
    settling_time_s = None
    for i, t in enumerate(times):
        if all(abs(e) <= position_tolerance_rad for e in errors[i:]) and all(
            abs(v) <= velocity_tolerance_rad_s for v in velocities[i:]
        ):
            settling_time_s = t
            break
    tail_start = max(0, int(0.75 * len(positions)))
    tail = positions[tail_start:] if positions else []
    tail_peak_to_peak_rad = (max(tail) - min(tail)) if tail else 0.0
    max_abs_velocity = max((abs(v) for v in velocities), default=0.0)
    velocity_limit_fraction = None
    if velocity_limit_rad_s and math.isfinite(velocity_limit_rad_s) and velocity_limit_rad_s > 0:
        velocity_limit_fraction = max_abs_velocity / velocity_limit_rad_s
    return {
        "start_rad": start,
        "target_rad": target,
        "command_delta_rad": delta,
        "final_rad": positions[-1] if positions else start,
        "final_error_rad": errors[-1] if errors else delta,
        "max_abs_error_rad": max((abs(e) for e in errors), default=0.0),
        "overshoot_rad": overshoot_rad,
        "overshoot_pct": overshoot_pct,
        "rise_time_90_s": rise_time_s,
        "settling_time_s": settling_time_s,
        "tail_peak_to_peak_rad": tail_peak_to_peak_rad,
        "max_abs_velocity_rad_s": max_abs_velocity,
        "velocity_limit_fraction": velocity_limit_fraction,
    }


def recommend_pd(
    joint_name: str,
    metrics: dict[str, Any],
    gains: dict[str, float | None],
    *,
    target_rise_time_s: float,
    position_tolerance_rad: float,
    overshoot_target_fraction: float = 0.01,
) -> dict[str, Any]:
    kp = float(gains["stiffness"] or 0.0)
    kd = float(gains["damping"] or 0.0)
    new_kp = kp
    new_kd = kd
    reasons: list[str] = []
    overshoot_pct = float(metrics["overshoot_pct"])
    tail_pp = float(metrics["tail_peak_to_peak_rad"])
    rise_time = metrics["rise_time_90_s"]
    settling_time = metrics["settling_time_s"]
    final_error = abs(float(metrics["final_error_rad"]))
    velocity_fraction = metrics.get("velocity_limit_fraction")
    overshoot_target_fraction = max(0.0, float(overshoot_target_fraction))
    overshoot_target = max(overshoot_target_fraction, 1.0e-6)
    oscillatory = overshoot_pct > overshoot_target_fraction or tail_pp > max(0.012, 1.5 * position_tolerance_rad)
    slow = rise_time is None or float(rise_time) > target_rise_time_s
    if oscillatory:
        overshoot_ratio = overshoot_pct / overshoot_target
        factor = min(2.5, max(1.08, 1.0 + 0.2 * max(0.0, overshoot_ratio - 1.0) + tail_pp / 0.05))
        new_kd *= factor
        reasons.append(
            f"increase damping by {factor:.2f}x because overshoot={overshoot_pct:.1%} "
            f"exceeds the {overshoot_target_fraction:.1%} target or tail p-p={tail_pp:.4f} rad"
        )
        if settling_time is None and overshoot_pct > 0.25:
            new_kp *= 0.9
            reasons.append("reduce stiffness 10% because the joint did not settle and overshoot is large")
    elif slow:
        factor = 1.2
        new_kp *= factor
        new_kd *= math.sqrt(factor)
        reasons.append(
            f"increase stiffness by {factor:.2f}x because rise_time_90={rise_time} s is slower than "
            f"{target_rise_time_s:.3f} s"
        )
        reasons.append("scale damping with sqrt(stiffness) to roughly preserve damping ratio")
    else:
        reasons.append("keep gains: rise, overshoot, and tail oscillation are within tuning targets")
    if final_error > position_tolerance_rad and not oscillatory:
        if velocity_fraction is not None and velocity_fraction < 0.35:
            new_kp *= 1.15
            new_kd *= math.sqrt(1.15)
            reasons.append("increase stiffness 15% because final error remains high while velocity usage is low")
        else:
            reasons.append(
                "final error remains high, but velocity usage is not low; inspect effort saturation before increasing "
                "stiffness"
            )
    return {
        "joint": joint_name,
        "current_stiffness": kp,
        "current_damping": kd,
        "recommended_stiffness": round(new_kp, 6),
        "recommended_damping": round(new_kd, 6),
        "changed": abs(new_kp - kp) > 1e-9 or abs(new_kd - kd) > 1e-9,
        "reasons": reasons,
    }


def plot_tracked_csv_by_joint(
    csv_path: Path,
    *,
    output_dir: Path | None = None,
    group_column: str = "joint",
    x_column: str = "time_s",
    image_format: str = "png",
    overshoot_by_group: dict[str, float] | None = None,
    rise_time_by_group: dict[str, float | None] | None = None,
    metrics_by_group: dict[str, dict[str, Any]] | None = None,
) -> dict[str, Any]:
    """Plot CSV-tracked joint values once per joint/group.

    The diagnostic CSVs are intentionally flat. This helper keeps plotting
    independent of a specific robot by detecting target/position/velocity
    columns at runtime and grouping by the ``joint`` column when present.
    """

    csv_path = Path(csv_path)
    if output_dir is None:
        output_dir = csv_path.parent / f"{csv_path.stem}_plots"
    output_dir = Path(output_dir)
    image_format = str(image_format or "png").lstrip(".")
    rows, fieldnames = _read_csv_rows(csv_path)
    if not rows:
        return {"enabled": True, "csv": str(csv_path), "warning": "CSV contains no data rows"}

    try:
        import matplotlib

        logging.getLogger("matplotlib").setLevel(logging.WARNING)
        if hasattr(matplotlib, "set_loglevel"):
            matplotlib.set_loglevel("warning")
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:  # noqa: BLE001  # pragma: no cover - optional dependency
        return {"enabled": True, "csv": str(csv_path), "error": f"matplotlib unavailable: {exc}"}

    groups = _group_csv_rows(rows, group_column)
    output_dir.mkdir(parents=True, exist_ok=True)
    joint_plots: dict[str, str] = {}
    overshoot_pct_by_group: dict[str, float | None] = {}
    plotted_metrics: dict[str, dict[str, float | None]] = {}
    skipped: dict[str, str] = {}
    for group_name, group_rows in groups.items():
        numeric_columns = [
            column
            for column in _numeric_columns(group_rows, fieldnames, exclude={group_column, x_column})
            if "error" not in column.lower()
        ]
        target_column = _first_numeric_column(group_rows, fieldnames, ("target", "goal", "command"), exclude=())
        position_column = _first_numeric_column(
            group_rows,
            fieldnames,
            ("position", "actual", "actuator"),
            exclude=("velocity", "error", "target", "goal", "command"),
        )
        velocity_column = _first_numeric_column(group_rows, fieldnames, ("velocity",), exclude=("error",))
        tracked_columns = _tracked_plot_columns(numeric_columns, target_column, position_column, velocity_column)
        if not tracked_columns:
            skipped[group_name] = "no numeric tracked columns"
            continue
        x_values, x_label = _csv_x_values(group_rows, x_column)
        if position_column and target_column:
            axes_count = 1 + (1 if velocity_column else 0)
            fig_height = 5.0 if axes_count == 2 else 3.2
            fig, axes = plt.subplots(axes_count, 1, figsize=(10.0, fig_height), sharex=True)
            axes = [axes] if axes_count == 1 else list(axes)
            target_values = [_to_float(row.get(target_column), default=float("nan")) for row in group_rows]
            position_values = [_to_float(row.get(position_column), default=float("nan")) for row in group_rows]
            group_metrics = _group_metrics(group_name, metrics_by_group)
            overshoot_pct = _overshoot_percent_for_group(
                group_name,
                position_values,
                target_values,
                overshoot_by_group,
                allow_trace_estimate=x_label != "sample",
            )
            overshoot_pct_by_group[group_name] = overshoot_pct
            axes[0].plot(x_values, target_values, linewidth=1.4, linestyle="--", label=_friendly_label(target_column))
            position_label = _friendly_label(position_column)
            position_details = []
            if overshoot_pct is not None:
                position_details.append(f"overshoot {overshoot_pct:.2f}%")
            rise_time = _metric_for_group(group_name, group_metrics, "rise_time_90_s", rise_time_by_group)
            if rise_time is not None:
                position_details.append(f"rise {rise_time:.3f}s")
            if position_details:
                position_label = f"{position_label} ({', '.join(position_details)})"
            tolerance = _metric_for_group(group_name, group_metrics, "position_tolerance_rad")
            if tolerance is not None and tolerance > 0:
                axes[0].fill_between(
                    x_values,
                    [target - tolerance for target in target_values],
                    [target + tolerance for target in target_values],
                    color="C0",
                    alpha=0.10,
                    linewidth=0,
                    label=f"target +/- {tolerance:.4g} rad",
                )
            axes[0].plot(x_values, position_values, linewidth=1.6, label=position_label)
            settling_time = _metric_for_group(group_name, group_metrics, "settling_time_s")
            if rise_time is not None:
                axes[0].axvline(rise_time, color="0.35", linestyle=":", linewidth=1.0, label="rise time")
            if settling_time is not None:
                axes[0].axvline(settling_time, color="0.25", linestyle="-.", linewidth=1.0, label="settling time")
            axes[0].set_ylabel(_position_ylabel(target_column, position_column))
            axes[0].grid(True, alpha=0.25)
            final_error = _metric_for_group(group_name, group_metrics, "final_error_rad")
            tail_pp = _metric_for_group(group_name, group_metrics, "tail_peak_to_peak_rad")
            _add_stats_box(
                axes[0],
                [
                    _format_time_stat("settle", settling_time),
                    _format_value_stat("final err", final_error, "rad"),
                    _format_value_stat("tail p-p", tail_pp, "rad"),
                ],
            )
            axes[0].legend(loc="best")
            if velocity_column:
                velocity_values = [_to_float(row.get(velocity_column), default=float("nan")) for row in group_rows]
                max_velocity = _metric_for_group(group_name, group_metrics, "max_abs_velocity_rad_s")
                velocity_fraction = _metric_for_group(group_name, group_metrics, "velocity_limit_fraction")
                velocity_label = _velocity_label(velocity_column, max_velocity, velocity_fraction)
                axes[1].plot(x_values, velocity_values, linewidth=1.4, label=velocity_label)
                axes[1].set_ylabel(_axis_ylabel(velocity_column))
                axes[1].grid(True, alpha=0.25)
                axes[1].legend(loc="best")
            plotted_metrics[group_name] = {
                "overshoot_pct": overshoot_pct,
                "rise_time_s": rise_time,
                "settling_time_s": settling_time,
                "final_error_rad": final_error,
                "tail_peak_to_peak_rad": tail_pp,
                "max_abs_velocity_rad_s": _metric_for_group(group_name, group_metrics, "max_abs_velocity_rad_s"),
                "velocity_limit_fraction": _metric_for_group(group_name, group_metrics, "velocity_limit_fraction"),
                "position_tolerance_rad": tolerance,
            }
        else:
            fig_height = max(2.4, min(14.0, 2.2 * len(tracked_columns)))
            fig, axes = plt.subplots(len(tracked_columns), 1, figsize=(10.0, fig_height), sharex=True)
            axes = [axes] if len(tracked_columns) == 1 else list(axes)
            overshoot_pct_by_group[group_name] = None
            plotted_metrics[group_name] = {}
            for ax, column in zip(axes, tracked_columns, strict=False):
                y_values = [_to_float(row.get(column), default=float("nan")) for row in group_rows]
                ax.plot(x_values, y_values, linewidth=1.5, label=_friendly_label(column))
                ax.set_ylabel(_axis_ylabel(column))
                ax.grid(True, alpha=0.25)
                ax.legend(loc="best")
        axes[-1].set_xlabel(_axis_ylabel(x_label))
        fig.suptitle(f"{csv_path.stem}: {group_name}")
        fig.tight_layout()
        plot_path = output_dir / f"{_safe_filename(group_name)}.{image_format}"
        fig.savefig(plot_path, dpi=140)
        plt.close(fig)
        joint_plots[group_name] = str(plot_path)
    return {
        "enabled": True,
        "csv": str(csv_path),
        "output_dir": str(output_dir),
        "group_column": group_column if group_column in fieldnames else None,
        "x_column": x_column if x_column in fieldnames else None,
        "joint_plots": joint_plots,
        "overshoot_pct": overshoot_pct_by_group,
        "rise_time_s": {
            group_name: _metric_for_group(
                group_name, _group_metrics(group_name, metrics_by_group), "rise_time_90_s", rise_time_by_group
            )
            for group_name in groups
        },
        "plotted_metrics": plotted_metrics,
        "skipped": skipped,
    }


def write_usd_drive_overlay(
    robot: Any,
    joint_gains: dict[str, dict[str, float | None]],
    output_path: Path,
    *,
    solver_position_iterations: int | None = None,
    solver_velocity_iterations: int | None = None,
    articulation_prim_paths: list[str] | None = None,
    include_joint_friction: bool = True,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Create a USDA override layer with tuned joint-drive values.

    The layer sublayers the loaded robot USD and authors stronger opinions for
    selected joint drive attributes. It does not modify the source USD.
    """

    metadata = load_robot_usd_metadata(robot)
    if metadata.load_error:
        raise RuntimeError(metadata.load_error)
    if not metadata.usd_path:
        raise RuntimeError("robot USD path is unavailable")

    try:
        from pxr import PhysxSchema, Sdf, Usd, UsdPhysics
    except ModuleNotFoundError as exc:
        raise RuntimeError(f"pxr modules unavailable: {exc}") from exc

    output_path = Path(output_path).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(f"USD overlay already exists: {output_path}")
        output_path.unlink()

    stage = Usd.Stage.CreateNew(str(output_path))
    root_layer = stage.GetRootLayer()
    root_layer.subLayerPaths.append(_relative_asset_path(Path(metadata.usd_path), output_path.parent))
    if metadata.default_prim:
        default_prim = stage.OverridePrim(Sdf.Path(metadata.default_prim))
        stage.SetDefaultPrim(default_prim)

    authored: dict[str, dict[str, float | None]] = {}
    missing_joints: list[str] = []
    for joint_name, gains in joint_gains.items():
        joint = metadata.joints.get(joint_name)
        if joint is None or joint.drive_axis is None:
            missing_joints.append(joint_name)
            continue
        prim = stage.OverridePrim(Sdf.Path(joint.path))
        drive = UsdPhysics.DriveAPI.Apply(prim, joint.drive_axis)
        authored[joint_name] = {}
        for field, attr_getter in (
            ("stiffness", drive.CreateStiffnessAttr),
            ("damping", drive.CreateDampingAttr),
            ("max_force", drive.CreateMaxForceAttr),
        ):
            value = gains.get(field)
            if value is not None:
                attr_getter().Set(float(value))
                authored[joint_name][field] = float(value)
        physx_joint = PhysxSchema.PhysxJointAPI.Apply(prim)
        if include_joint_friction:
            friction = gains.get("joint_friction", gains.get("friction"))
            if friction is not None:
                physx_joint.CreateJointFrictionAttr().Set(float(friction))
                authored[joint_name]["joint_friction"] = float(friction)
        max_joint_velocity = gains.get("max_joint_velocity")
        max_joint_velocity_sim = gains.get("max_joint_velocity_sim")
        if max_joint_velocity_sim is not None:
            sim_value = float(max_joint_velocity_sim)
            authored_value = runtime_velocity_limit_to_usd(robot, joint_name, sim_value)
            physx_joint.CreateMaxJointVelocityAttr().Set(authored_value)
            authored[joint_name]["max_joint_velocity"] = authored_value
            authored[joint_name]["max_joint_velocity_sim"] = sim_value
        elif max_joint_velocity is not None:
            physx_joint.CreateMaxJointVelocityAttr().Set(float(max_joint_velocity))
            authored[joint_name]["max_joint_velocity"] = float(max_joint_velocity)

    solver_attrs: dict[str, dict[str, int]] = {}
    roots = articulation_prim_paths if articulation_prim_paths is not None else metadata.articulation_roots
    if solver_position_iterations is not None or solver_velocity_iterations is not None:
        for root_path in roots:
            prim = stage.OverridePrim(Sdf.Path(root_path))
            PhysxSchema.PhysxArticulationAPI.Apply(prim)
            solver_attrs[root_path] = {}
            if solver_position_iterations is not None:
                value = int(solver_position_iterations)
                prim.CreateAttribute("physxArticulation:solverPositionIterationCount", Sdf.ValueTypeNames.Int).Set(
                    value
                )
                solver_attrs[root_path]["solver_position_iterations"] = value
            if solver_velocity_iterations is not None:
                value = int(solver_velocity_iterations)
                prim.CreateAttribute("physxArticulation:solverVelocityIterationCount", Sdf.ValueTypeNames.Int).Set(
                    value
                )
                solver_attrs[root_path]["solver_velocity_iterations"] = value

    root_layer.Save()
    return {
        "path": str(output_path),
        "source_usd": metadata.usd_path,
        "sublayer": root_layer.subLayerPaths[0],
        "authored_joints": authored,
        "missing_joints": missing_joints,
        "solver_overrides": solver_attrs,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_csv(path: Path, rows: list[dict[str, Any]], fieldnames: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def result_paths(name: str) -> tuple[Path, Path]:
    rd = run_dir()
    return rd / f"{name}.json", rd / f"{name}.csv"


def _read_csv_rows(csv_path: Path) -> tuple[list[dict[str, str]], list[str]]:
    with Path(csv_path).open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        return list(reader), list(reader.fieldnames or [])


def _group_csv_rows(rows: list[dict[str, str]], group_column: str) -> dict[str, list[dict[str, str]]]:
    if rows and group_column in rows[0]:
        groups: dict[str, list[dict[str, str]]] = {}
        for row in rows:
            groups.setdefault(str(row.get(group_column) or "ungrouped"), []).append(row)
        return groups
    return {"all": rows}


def _numeric_columns(rows: list[dict[str, str]], fieldnames: list[str], *, exclude: set[str]) -> list[str]:
    out: list[str] = []
    for field in fieldnames:
        if field in exclude:
            continue
        values = [row.get(field) for row in rows]
        if any(_is_float(value) for value in values):
            out.append(field)
    return out


def _first_numeric_column(
    rows: list[dict[str, str]], fieldnames: list[str], include: tuple[str, ...], *, exclude: tuple[str, ...]
) -> str | None:
    for field in fieldnames:
        lowered = field.lower()
        if not any(token in lowered for token in include):
            continue
        if any(token in lowered for token in exclude):
            continue
        if any(_is_float(row.get(field)) for row in rows):
            return field
    return None


def _tracked_plot_columns(
    numeric_columns: list[str], target_column: str | None, position_column: str | None, velocity_column: str | None
) -> list[str]:
    if target_column and position_column:
        out = [target_column, position_column]
        if velocity_column:
            out.append(velocity_column)
        return out
    preferred = [column for column in (target_column, position_column, velocity_column) if column]
    return preferred or numeric_columns


def _csv_x_values(rows: list[dict[str, str]], x_column: str) -> tuple[list[float], str]:
    if rows and x_column in rows[0] and all(_is_float(row.get(x_column)) for row in rows):
        return [_to_float(row.get(x_column), default=float(index)) for index, row in enumerate(rows)], x_column
    return [float(index) for index, _ in enumerate(rows)], "sample"


def _is_float(value: Any) -> bool:
    try:
        float(value)
    except (TypeError, ValueError):
        return False
    return True


def _to_float(value: Any, *, default: float) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _overshoot_percent_for_group(
    group_name: str,
    position_values: list[float],
    target_values: list[float],
    overshoot_by_group: dict[str, float] | None,
    *,
    allow_trace_estimate: bool,
) -> float | None:
    if overshoot_by_group and group_name in overshoot_by_group:
        value = maybe_float(overshoot_by_group[group_name])
        if value is not None and math.isfinite(value):
            return 100.0 * value
    if not allow_trace_estimate:
        return None
    return _trace_overshoot_percent(position_values, target_values)


def _trace_overshoot_percent(position_values: list[float], target_values: list[float]) -> float | None:
    finite_positions = [value for value in position_values if math.isfinite(value)]
    finite_targets = [value for value in target_values if math.isfinite(value)]
    if not finite_positions or not finite_targets:
        return None
    start = finite_positions[0]
    target = finite_targets[-1]
    delta = target - start
    if abs(delta) < 1.0e-9:
        return 0.0
    progress = [(position - start) / delta for position in finite_positions]
    return max(0.0, max(progress, default=0.0) - 1.0) * 100.0


def _group_metrics(group_name: str, metrics_by_group: dict[str, dict[str, Any]] | None) -> dict[str, Any]:
    if not metrics_by_group:
        return {}
    metrics = metrics_by_group.get(group_name)
    return metrics if isinstance(metrics, dict) else {}


def _metric_for_group(
    group_name: str,
    metrics: dict[str, Any],
    key: str,
    fallback_by_group: dict[str, float | None] | None = None,
) -> float | None:
    if key in metrics:
        value = maybe_float(metrics[key])
        if value is not None and math.isfinite(value):
            return value
    if not fallback_by_group or group_name not in fallback_by_group:
        return None
    value = maybe_float(fallback_by_group[group_name])
    if value is None or not math.isfinite(value):
        return None
    return value


def _add_stats_box(ax: Any, lines: list[str | None]) -> None:
    text = "\n".join(line for line in lines if line)
    if not text:
        return
    ax.text(
        0.99,
        0.04,
        text,
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.5,
        bbox={"boxstyle": "round,pad=0.25", "facecolor": "white", "edgecolor": "0.75", "alpha": 0.82},
    )


def _format_time_stat(label: str, value: float | None) -> str | None:
    if value is None:
        return None
    return f"{label}: {value:.3f}s"


def _format_value_stat(label: str, value: float | None, unit: str) -> str | None:
    if value is None:
        return None
    return f"{label}: {value:.4g} {unit}"


def _velocity_label(column: str, max_velocity: float | None, velocity_fraction: float | None) -> str:
    details = []
    if max_velocity is not None:
        details.append(f"max {max_velocity:.3f} rad/s")
    if velocity_fraction is not None:
        details.append(f"limit {100.0 * velocity_fraction:.1f}%")
    label = _friendly_label(column)
    return f"{label} ({', '.join(details)})" if details else label


def _friendly_label(column: str) -> str:
    lowered = column.lower()
    if any(token in lowered for token in ("target", "goal", "command")):
        return "target position"
    if any(token in lowered for token in ("position", "actual", "actuator")):
        return "actuator position"
    if "velocity" in lowered:
        return "velocity"
    return column


def _position_ylabel(target_column: str, position_column: str) -> str:
    unit = _unit_from_column(target_column) or _unit_from_column(position_column)
    return f"position ({unit})" if unit else "position"


def _axis_ylabel(column: str) -> str:
    lowered = column.lower()
    if lowered == "time_s":
        return "time (s)"
    if "velocity" in lowered:
        unit = _unit_from_column(column)
        if unit == "rad_s":
            return "velocity (rad/s)"
        if unit == "m_s":
            return "velocity (m/s)"
        return "velocity"
    unit = _unit_from_column(column)
    if unit:
        return f"{column.removesuffix('_' + unit)} ({unit.replace('_s', '/s')})"
    return column


def _unit_from_column(column: str) -> str | None:
    lowered = column.lower()
    if lowered.endswith("_rad_s") or "_rad_s_" in lowered:
        return "rad_s"
    if lowered.endswith("_m_s") or "_m_s_" in lowered:
        return "m_s"
    if lowered.endswith("_rad") or "_rad_" in lowered:
        return "rad"
    if lowered.endswith("_m") or "_m_" in lowered:
        return "m"
    return None


def _safe_filename(value: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("._")
    return safe or "plot"


def _relative_asset_path(source_path: Path, output_dir: Path) -> str:
    try:
        return os.path.relpath(str(source_path.resolve()), str(output_dir.resolve()))
    except ValueError:
        return str(source_path)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Freeze navigation fixture geometry or record its legacy reference at the approved pins.

Run from the Workflow root with the reference environment for ``reference``.
This is a qualification utility; it changes neither workflow's selected solver.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]


def load_parity(numerical_root):
    path = numerical_root / "scripts/parity/catheter.py"
    spec = importlib.util.spec_from_file_location("catheter_parity", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def geometry(config, fixture):
    from i4h_common.centerline import sample_polyline

    g, r = config["geometry"], config["rod"]
    arc = np.linspace(0.0, g["route_length"], round(g["route_length"] / g["route_spacing"]) + 1)
    if fixture.get("geometry") == "straight":
        route = np.column_stack((arc, np.zeros_like(arc), np.zeros_like(arc)))
    else:
        radius = g["curve_radius"]
        route = np.column_stack(
            (radius * np.sin(arc / radius), radius * (1.0 - np.cos(arc / radius)), np.zeros_like(arc))
        )
    route += np.asarray(g["start"])
    points = np.asarray(sample_polyline(route, np.linspace(0.0, r["length"], r["segments"] + 1)), np.float32)
    points[:, 2] += fixture.get("initial_lateral_offset", 0.0)
    wall_arc = np.linspace(0.0, g["route_length"], round(g["route_length"] / g["wall_spacing"]) + 1)
    wall = np.asarray(sample_polyline(route, wall_arc), np.float32)
    edges = np.column_stack((np.arange(len(wall) - 1), np.arange(1, len(wall)))).astype(np.int32)
    if fixture.get("geometry") != "straight":
        root = round(g["branch_at_arc"] / g["wall_spacing"])
        tangent_angle = g["branch_at_arc"] / g["curve_radius"] + g["branch_angle_rad"]
        branch_arc = np.linspace(0.0, g["branch_length"], round(g["branch_length"] / g["wall_spacing"]) + 1)[1:]
        branch = wall[root] + branch_arc[:, None] * np.array([np.cos(tangent_angle), np.sin(tangent_angle), 0.0])
        indices = np.r_[root, np.arange(len(wall), len(wall) + len(branch))]
        edges = np.concatenate((edges, np.column_stack((indices[:-1], indices[1:]))))
        wall = np.concatenate((wall, branch.astype(np.float32)))
    return route.astype(np.float32), points, wall, edges


def wall_settings(config):
    settings = config["wall"].copy()
    dynamics = {
        key: settings.pop(key)
        for key in (
            "node_mass",
            "segment_inertia",
            "bend_stiffness",
            "twist_stiffness",
            "stretch_stiffness",
            "root_locked",
            "endpoints_locked",
            "iterations",
            "linear_damping",
            "angular_damping",
            "gravity",
        )
    }
    settings.pop("contact_coupling_iterations")
    settings.pop("edge_samples")
    return {"dynamics": dynamics, **settings}


def root_driver(config, fixture, route):
    import torch

    from i4h_arena.medical.catheter_drive import (
        FlangeMountedIntroducer,
        IntroducerDriveSpec,
        LumenClamp,
        RouteRailedIntroducer,
        quat_to_w_first,
        quat_to_xyzw,
    )

    dt = config["rod"]["dt"]
    path = torch.from_numpy(route)
    rates = torch.tensor(fixture["rates"][:2], dtype=torch.float32).reshape(1, 2)
    if fixture["drive"] == "route":
        drive = RouteRailedIntroducer(path, 1)
    else:
        drive = FlangeMountedIntroducer(IntroducerDriveSpec(), 1)
    clamp = LumenClamp(path, torch.full((len(path),), config["geometry"]["lumen_radius"]), config["rod"]["radius"])

    def root_pose(step, points, frames):
        spent, spin = drive.advance(rates[:, 0], rates[:, 1], dt)
        if fixture["drive"] == "route":
            position, orientation = drive.root_target()
        else:
            flange = torch.tensor(config["geometry"]["start"]) + torch.tensor(fixture["flange_velocity"]) * (step * dt)
            transport = drive.transport(flange.reshape(1, 3))
            # Matches ArmDrivenCatheterAction: flange position transport and
            # current shaft tangent feed; flange orientation is not added again.
            q = torch.as_tensor(np.asarray(points).copy()).reshape(1, -1, 3)
            f = torch.as_tensor(np.asarray(frames).copy()).reshape(1, -1, 4)
            position, orientation = drive.root_target(
                q[:, 0], quat_to_w_first(f[:, 0]), q[:, 1] - q[:, 0], transport, spent, spin, dt
            )
            position = clamp.project(position)
        p, q = position.numpy().astype(np.float32), quat_to_xyzw(orientation).numpy().astype(np.float32)
        root_pose.commands.append(np.concatenate((p[0], q[0])))
        return p, q

    root_pose.commands = []
    return root_pose


def arrival_decisions(points, target, gates):
    """Use the Workflow's actual success predicate at the existing control cadence."""
    from types import SimpleNamespace

    import torch

    from i4h_arena.medical.navigation_goal import reached_navigation_target

    data = SimpleNamespace(positions_world_m=None)
    env = SimpleNamespace(scene={"catheter": SimpleNamespace(data=data)}, num_envs=1, device="cpu")
    decisions = []
    for frame in points[3::4]:
        data.positions_world_m = torch.from_numpy(frame[None].copy())
        decisions.append(
            bool(
                reached_navigation_target(
                    env, target, gates["arrival_tolerance_m"], gates["arrival_hold_control_steps"]
                )[0]
            )
        )
    return np.asarray(decisions)


def escapes(points, wall, edges, free_radius):
    """Number of nodes outside the live linear lumen per physics sample."""
    masks, branches = [], []
    for frame, displacement in zip(points, wall, strict=True):
        a, b = displacement[edges[:, 0]], displacement[edges[:, 1]]
        span = b - a
        t = np.clip(
            np.sum((frame[:, None] - a) * span, axis=-1) / np.maximum(np.sum(span * span, axis=-1), 1e-20), 0.0, 1.0
        )
        distances = np.linalg.norm(frame[:, None] - a - t[..., None] * span, axis=-1)
        masks.append(distances.min(axis=1) > free_radius)
        branches.append(int(np.argmin(distances[-1])))
    return np.asarray(masks), np.asarray(branches)


def compare_navigation(run_dir, config):
    inputs = dict(np.load(run_dir / "navigation-inputs.npz"))
    if json.loads(str(inputs["config"])) != config:
        raise ValueError("current fixture settings or thresholds differ from the frozen inputs")
    ref = dict(np.load(run_dir / "navigation-reference.npz"))
    ours = dict(np.load(run_dir / "navigation-numerical.npz"))
    for archive in (ref, ours):
        if not bool(archive.get("complete", False)) or str(archive["fixture_sha256"]) != str(inputs["fixture_sha256"]):
            raise ValueError("incomplete or mismatched navigation qualification")
    if not ref.get("workflow_sources") or str(ref["workflow_sources"]) != str(ours.get("workflow_sources")):
        raise ValueError("reference and numerical runs used different Workflow control sources")
    rows = []

    def row(name, metric, value, bound):
        rows.append(
            {
                "fixture": name,
                "metric": metric,
                "value": float(value),
                "bound": float(bound),
                "pass": bool(np.isfinite(value) and value <= bound),
            }
        )

    for fixture in config["fixtures"]:
        name = fixture["name"]
        prefix = name + "/"
        for key in (
            "positions",
            "orientations",
            "velocities",
            "angular_velocities",
            "reaction",
            "vessel_displacement",
            "commands",
            "root_commands",
            "time",
        ):
            if ref[prefix + key].shape != ours[prefix + key].shape or not all(
                np.isfinite(archive[prefix + key]).all() for archive in (ref, ours)
            ):
                raise ValueError(f"invalid navigation trace: {prefix + key}")
        for key in ("commands", "time"):
            if not np.array_equal(ref[prefix + key], ours[prefix + key]):
                raise ValueError(f"navigation inputs differ: {prefix + key}")
        a, b = ref[prefix + "positions"], ours[prefix + "positions"]
        row(
            name,
            "tip RMS / L",
            np.sqrt(np.mean(np.sum((a[:, -1] - b[:, -1]) ** 2, axis=1))) / config["rod"]["length"],
            config["gates"]["tip_rms_over_length"],
        )
        for key, gate, columns, label in (
            ("reaction", "reaction_relative_l2", slice(0, 3), "force"),
            ("reaction", "reaction_relative_l2", slice(3, 6), "moment"),
            ("vessel_displacement", "wall_displacement_relative_l2", slice(None), "vessel_displacement"),
        ):
            x, y = ref[prefix + key][..., columns], ours[prefix + key][..., columns]
            norm = np.linalg.norm(x)
            if norm == 0.0:
                row(name, label + " zero-response error", np.linalg.norm(y), config["gates"]["zero_response_atol"])
            else:
                row(name, label + " relative L2", np.linalg.norm(x - y) / norm, config["gates"][gate])
        if fixture["wall"]:
            rest, edges = inputs[prefix + "wall"], inputs[prefix + "wall_edges"]
            radius = config["geometry"]["lumen_radius"] - config["rod"]["radius"]
            counts = [
                escapes(
                    archive[prefix + "positions"], rest[None] + archive[prefix + "vessel_displacement"], edges, radius
                )
                for archive in (ref, ours)
            ]
            row(
                name,
                "new wall escapes",
                np.count_nonzero(counts[1][0] & ~counts[0][0], axis=1).max(),
                config["gates"]["new_wall_escapes"],
            )
            main_edges = round(config["geometry"]["route_length"] / config["geometry"]["wall_spacing"])
            new_branch = (counts[1][1] >= main_edges) & (counts[0][1] < main_edges)
            row(
                name,
                "new wrong-branch frames",
                np.count_nonzero(new_branch),
                config["gates"]["wrong_branch_transitions"],
            )
        if name.startswith("arrival_"):
            target = (
                inputs[prefix + "positions"][-1] if fixture["goal_at_initial_tip"] else inputs[prefix + "route"][-1]
            )
            expected = config["gates"][name]
            for engine, archive in (("reference", ref), ("numerical", ours)):
                decisions = arrival_decisions(archive[prefix + "positions"], target, config["gates"])
                row(name, engine + " final arrival decision errors", float(bool(decisions[-1]) != expected), 0)
    result = {
        "measured_gates_passed": all(r["pass"] for r in rows),
        "promotion_passed": False,
        "rows": rows,
        "unqualified": ["wrong-branch transition classifier", "full MuJoCo-Warp arm coupling", "native lifecycle"],
    }
    (run_dir / "navigation-report.json").write_text(json.dumps(result, indent=2) + "\n")
    for r in rows:
        print(("PASS" if r["pass"] else "FAIL"), r["fixture"], r["metric"], r["value"], "bound", r["bound"])
    return 0 if result["measured_gates_passed"] else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("freeze", "reference", "numerical", "compare"))
    parser.add_argument(
        "--numerical-root", type=Path, required=True, help="Numerical Sim workspace, containing scripts/parity"
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    args = parser.parse_args()
    sys.path[:0] = [str(ROOT / "arena"), str(ROOT / "common")]
    from i4h_arena.medical.catheter_initialization import rod_frames_along_polyline

    fixture_path = args.numerical_root / "scripts/parity/navigation-fixtures.json"
    config = json.loads(fixture_path.read_text())
    if args.command == "compare":
        return compare_navigation(args.run_dir, config)
    revision = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip()
    ancestor = subprocess.run(["git", "merge-base", "--is-ancestor", config["workflow_revision"], revision], cwd=ROOT)
    if ancestor.returncode:
        raise ValueError(f"expected workflow base {config['workflow_revision']}, got {revision}")
    args.run_dir.mkdir(parents=True, exist_ok=True)
    inputs = {
        "fixture_sha256": hashlib.sha256(fixture_path.read_bytes()).hexdigest(),
        "config": json.dumps(config),
        "workflow_revision": subprocess.check_output(
            ["git", "rev-parse", config["workflow_revision"]], cwd=ROOT, text=True
        ).strip(),
    }
    for fixture in config["fixtures"]:
        name = fixture["name"]
        route, points, wall, edges = geometry(config, fixture)
        inputs.update(
            {
                name + "/route": route,
                name + "/positions": points,
                name + "/wall": wall,
                name + "/orientations": rod_frames_along_polyline(points),
                name + "/wall_edges": edges,
            }
        )
    target = args.run_dir / "navigation-inputs.npz"
    if target.exists():
        with np.load(target) as saved:
            if set(saved.files) != set(inputs) or any(not np.array_equal(saved[k], v) for k, v in inputs.items()):
                raise ValueError("frozen navigation inputs differ; use a new canonical run directory")
    else:
        np.savez(target, **inputs)
    if args.command == "freeze":
        print(target)
        return 0
    parity = load_parity(args.numerical_root)
    output = {
        "schema": 1,
        "complete": False,
        "workflow_revision": revision,
        "runner_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "workflow_sources": json.dumps(
            {
                name: hashlib.sha256((ROOT / "arena/i4h_arena/medical" / name).read_bytes()).hexdigest()
                for name in ("catheter_drive.py", "catheter_initialization.py", "navigation_goal.py")
            }
        ),
        "source": json.dumps(parity.source_identity("reference" if args.command == "reference" else "ours")),
        "fixture_sha256": inputs["fixture_sha256"],
    }
    rod = parity.Rod(**config["rod"])
    for fixture in config["fixtures"]:
        name = fixture["name"]
        print(f"{args.command} fixture {name}: starting", flush=True)
        route, points, wall = (inputs[name + "/" + key] for key in ("route", "positions", "wall"))

        def command(k, fixture=fixture):
            push, rotation, bend = fixture["rates"]
            return push, rotation, np.clip(bend * (k + 1) * rod.dt, -1.5, 1.5)

        driver = root_driver(config, fixture, route)
        run = parity.run_reference if args.command == "reference" else parity.run_ours
        trace = run(
            rod,
            fixture.get("gravity", (0.0, 0.0, 0.0)),
            round(fixture["seconds"] / rod.dt),
            command,
            lumen={"points": wall, "edges": inputs[name + "/wall_edges"]} if fixture["wall"] else None,
            initial_positions=points,
            initial_frames=inputs[name + "/orientations"],
            root_pose=driver,
            vessel_settings=wall_settings(config) if fixture["wall"] else None,
            sample_stride=1,
        )
        trace["root_commands"] = np.asarray(driver.commands)
        output.update({name + "/" + key: value for key, value in trace.items()})
        np.savez(args.run_dir / f"navigation-{args.command}.npz", **output)
        print(f"{args.command} fixture {name}: tip {trace['positions'][-1, -1]}", flush=True)
    output["complete"] = True
    np.savez(args.run_dir / f"navigation-{args.command}.npz", **output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

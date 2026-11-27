#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Summarize retained liver qualification, including the stopped PPO campaign."""

from __future__ import annotations

import argparse
import json
import subprocess
from datetime import UTC, datetime
from importlib import metadata
from pathlib import Path

import numpy as np

from i4h_rl.liver_experiment import digest, write_json


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.evidence.resolve()

    def read(name):
        return json.loads((folder / name).read_text())

    labels = ("controller", "bc", "ppo_stopped")
    evaluations = {label: read(f"{label}_reserved_100.json") for label in labels}
    assert all(row["objective"] == "full_cycle" and row["episodes"] == 100 for row in evaluations.values())
    assert all(evaluations[label]["starts"] == evaluations["controller"]["starts"] for label in labels)
    metrics = (
        "actual_lift_m",
        "max_lift_m",
        "minimum_retained_fraction",
        "max_radial_residual_m",
        "release_latency_s",
        "max_attachment_drift_m",
        "max_inverted_tets",
    )
    comparison = {}
    for label, evaluation in evaluations.items():
        physical = {}
        for metric in metrics:
            values = [row[metric] for row in evaluation["results"] if row[metric] is not None]
            physical[metric] = (
                None
                if not values
                else dict(count=len(values), minimum=min(values), mean=float(np.mean(values)), maximum=max(values))
            )
        comparison[label] = {
            key: evaluation[key]
            for key in ("episodes", "successes", "success_rate", "wilson_95", "per_target", "failures")
        }
        comparison[label]["physical"] = physical
    write_json(folder / "comparison.json", dict(identical_100_starts=True, policies=comparison))
    source_names = subprocess.check_output(
        ["git", "ls-files", "--modified", "--others", "--exclude-standard"], cwd=root, text=True
    ).splitlines()
    source_paths = [
        root / name
        for name in source_names
        if not name.startswith("outputs/")
        and Path(name).suffix in (".py", ".json", ".yaml", ".toml", ".sh", ".patch", ".lock")
    ]
    from i4h_common.paths import numerical_source

    num = numerical_source(root).parent
    # Setup applies the Isaac Lab 3 builder patch to the solver checkout that holds numerical_sim,
    # so only numerical_sim itself must be clean; the solver patch is recorded separately.
    live_diff = subprocess.check_output(["git", "diff", "--", "numerical_sim"], cwd=num)
    solver_diff = subprocess.check_output(["git", "diff", "--", ".", ":(exclude)numerical_sim"], cwd=num)
    import hashlib

    assert not live_diff, "Merged Numerical Sim must be unpatched"

    def git_head(directory):
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=directory, text=True).strip()

    provenance = dict(
        created_utc=datetime.now(UTC).isoformat(),
        workflows_head=git_head(root),
        numerical_head=git_head(num),
        numerical_patch_sha256=None,
        solver_patch_sha256=hashlib.sha256(solver_diff).hexdigest(),
        numerical_has_shared_object_alternates=(num / ".git/objects/info/alternates").exists(),
        isaaclab_head=git_head(root / "third_party/IsaacLab-3939445"),
        arena_head=git_head(root / "third_party/IsaacLab-Arena-67f5b19"),
        isaaclab_local_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root / "third_party/IsaacLab-3939445")
        ).hexdigest(),
        arena_local_diff_sha256=hashlib.sha256(
            subprocess.check_output(["git", "diff"], cwd=root / "third_party/IsaacLab-Arena-67f5b19")
        ).hexdigest(),
        wheels={dist.metadata["Name"]: dist.version for dist in metadata.distributions() if dist.metadata["Name"]},
        numerical_identity=json.loads((root / "outputs/liver_snapshots/manifest_16.json").read_text())["identity"],
        bc_checkpoint_sha256=digest(folder / "bc_best.pt"),
        stopped_ppo_checkpoint_sha256=digest(folder / "actor_validation/update_05/stopped_checkpoint.pt"),
        calibration_sha256=digest(root / "arena/i4h_arena/envcfg/liver_reach_retract/calibration.json"),
        source_hashes={str(path.relative_to(root)): digest(path) for path in source_paths},
        qualification_manifests={
            name: read(name)
            for name in (
                "qualification.json",
                "qualification_final.json",
                "reach_only/qualification.json",
                "reach_only/qualification_final.json",
                "grasped_retract/qualification.json",
                "grasped_retract/qualification_final.json",
            )
        },
    )
    write_json(folder / "provenance.json", provenance)
    lines = [
        "# Liver reach and retract integration results",
        "",
        (
            "The integration is implemented on Workflows `1208154b` with Numerical Sim "
            "`caff73b12a77abd6157771678ee0a57206771ac4`."
        ),
        (
            "Full-cycle PPO qualification **failed**: the prescribed deterministic guard stopped "
            "learning at actor update 5 after success fell from 20/20 to 12/20. The run was not "
            "resumed or adjusted. Separate curricula do not replace this failed full-cycle result."
        ),
        "",
        "## Matched reserved evaluation",
        "",
        (
            "Each policy received the same 100 reserved starts, 20 per target. Target, jitter, "
            "canonical environment row, initial tissue bytes and physical action hashes match "
            "exactly. These starts use seed 981304; validation uses 88201."
        ),
        "",
        "| Execution | Full-cycle success | 95% Wilson interval | Failures |",
        "|---|---:|---:|---|",
    ]
    for label, row in comparison.items():
        lo, hi = row["wilson_95"]
        lines.append(f"| {label} | {row['successes']}/100 | {lo:.1%}–{hi:.1%} | {row['failures'] or 'none'} |")
    lines += ["", "| Target | Controller | BC | Stopped PPO |", "|---|---:|---:|---:|"]
    for target in (252, 257, 255, 1688, 1098):
        values = []
        for label in labels:
            target_result = comparison[label]["per_target"][str(target)]
            lo, hi = target_result["wilson_95"]
            values.append(f"{target_result['successes']}/20 ({lo:.1%}–{hi:.1%})")
        lines.append(f"| {target} | {values[0]} | {values[1]} | {values[2]} |")

    lines += [
        "",
        (
            "Per-target cells show 95% Wilson intervals; per-episode terminal values/extrema are retained in "
            "`comparison.json` and `*_reserved_100.json`."
        ),
        "",
        (
            "| Execution | Peak lift range (mm) | Final lift mean after relaxation (mm) | Minimum "
            "retained fraction | Max release latency (ms) | Max attachment drift (mm) | Max "
            "inversions |"
        ),
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for label, row in comparison.items():
        p = row["physical"]
        lift = p["max_lift_m"]
        final = p["actual_lift_m"]
        retention = p["minimum_retained_fraction"]
        latency = p["release_latency_s"]
        lines.append(
            f"| {label} | {1000 * lift['minimum']:.3f}–{1000 * lift['maximum']:.3f} | "
            f"{1000 * final['mean']:.3f} | {retention['minimum']:.3f} | "
            f"{1000 * latency['maximum']:.3f} | {1000 * p['max_attachment_drift_m']['maximum']:.3f} | "
            f"{p['max_inverted_tets']['maximum']} |"
        )
    lines += [
        "",
        (
            "Extrema include failed episodes. Null acquisition/release measurements are excluded from"
            " those metric summaries, with denominators retained in JSON. Peak lift is distinct from "
            "final lift after release and relaxation. Success uses a +20 mm destination with 2 mm "
            "tolerance. Soft-attachment drift and inversions remain present; no mechanics were tuned."
        ),
        "",
        "## Qualification evidence",
        "",
        (
            "- Ordinary and snapshot reset isolation passed at 1, 8 and 16 environments, including "
            "live captured particle/sphere/rest bindings and probe latches. The selected world was in"
            " the middle of the batch for the grasped fixture. Peer continuation/task/probe/RNG "
            "arrays remained bitwise unchanged."
        ),
        (
            "- Complete-state recording covers numerical continuation, probe identities, Arena "
            "timers/extrema, held and previous commands, controller hysteresis, Lab action/episode "
            "history and per-world RNG. Incompatible identity/layout bundles fail explicitly."
        ),
        (
            "- Fresh-process initial observations matched exactly. Over the next control tick (two "
            "physics samples), maximum particle error across the three batch sizes was 0.778 "
            "micrometres, below the 10 micrometre continuation tolerance. This is not a claim of "
            "bitwise equality over an entire episode."
        ),
        (
            "- Action displacement is at most 1 mm per control tick, within float32 roundoff (largest"
            " measured 1.0000068 mm). Both physics samples are observed; repeated getters leave "
            "reducer state unchanged."
        ),
        (
            "- BC import and standard native PPO export matched all 144 frozen-input means and log "
            "probabilities exactly; analytic KL was zero. All four source/pre-update standard "
            "deviations were 0.02. Actor and critic running normalization are disabled."
        ),
        (
            "- Exact-BC zero-LR and critic-only updates preserved actor/distribution/normalization "
            "state. Rollback tests restore existing Adam moments. Runtime checkpoint resume restored "
            "the curriculum environment and completed one additional update."
        ),
        (
            "- The reference controller is command-identical to the pinned manual controller, while "
            "owning no objective phase, deadline or failure latch."
        ),
        (
            "- A visible controller Workflow completed 1/1 episodes; viewport captures show the live "
            "tissue, virtual grasper and target/destination markers. HDF5 replay completed 1/1. "
            "Standard exported stopped-PPO policy execution completed 1/1, agreeing with matched "
            "native execution on that start."
        ),
        (
            "- The BC single-environment seed-7341 attempt failed with retention loss in both "
            "Workflow and matched native evaluation. Both records are preserved. This is separate "
            "from the 16-environment reserved evaluation."
        ),
        (
            "- The broad regression suite passed 1377 tests with 3 skips. Subsequent focused "
            "lifecycle, controller-parity and resume tests also passed. Lint, lock consistency and "
            "patch reverse-apply checks passed."
        ),
        "",
        "## Learning disposition",
        "",
        (
            "Full-cycle: ten critic-only updates, then five attempted actor updates. Validation was "
            "20/20 at actor update 1 and 12/20 at update 5; the required stop prevented update 10. "
            "The mean-KL guard rejected multiple updates above 0.02. Source std and jaw-noise "
            "settings stayed fixed."
        ),
        "",
        (
            "Reach-only and grasped-retract each completed ten critic-only and ten actor updates. "
            "Deterministic validation was 20/20 at actor updates 1, 5 and 10 for each objective. "
            "Their exports carry curriculum identifiers and are rejected by the full-cycle Workflow."
        ),
        "",
        (
            "The stopped checkpoint and its lower reserved success are retained for review. A saved "
            "checkpoint or a successful single episode does not establish a successful full-cycle PPO"
            " training result."
        ),
        "",
        "## Artifact map",
        "",
        (
            "- `provenance.json`: exact dependency commits, installed wheel versions, local upstream "
            "diff hashes, asset/source hashes and checkpoint provenance."
        ),
        "- `isolation_*.json`, `grasped_isolation_*.json`, `fresh_replay_*.json`: batch qualification.",
        (
            "- `exported/`: BC inference export and frozen-input parity; `normal_export/`: export "
            "produced by the standard RL command and its parity check."
        ),
        (
            "- `actor_validation/update_05/stopped_checkpoint.pt`: failed full-cycle checkpoint, "
            "explicitly marked stopped. Do not resume it."
        ),
        (
            "- `reach_only/actor_validation/`, `grasped_retract/actor_validation/`: separate "
            "1/5/10-update deterministic checks."
        ),
        "- `visible_controller/`: normal Workflow HDF5, physical trace and viewport PNGs.",
        (
            "- `../../runs/liver_reach_retract/`: canonical launcher/trainer run directories and "
            "preserved failed attempts."
        ),
        "",
    ]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(lines))


if __name__ == "__main__":
    main()

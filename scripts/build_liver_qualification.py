#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build a PPO gate manifest from retained runtime and test evidence."""

import argparse
import json
import re
import subprocess
from importlib import metadata
from pathlib import Path

import i4h_physics
import i4h_surgical

from i4h_rl.liver_experiment import digest, write_json
from i4h_rl.liver_ppo import REQUIRED_GATES


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--tests", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--objective", choices=("full_cycle", "reach_only", "grasped_retract"), default="full_cycle")
    parser.add_argument("--baseline", type=Path)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    folder = args.evidence.resolve()

    def read(name):
        return json.loads((folder / name).read_text())

    numerical = root / "third_party/i4h-numerical-sim-caff73b"
    revision = subprocess.check_output(["git", "-C", str(numerical), "rev-parse", "HEAD"], text=True).strip()
    assert revision == "caff73b12a77abd6157771678ee0a57206771ac4"
    for module in (i4h_physics, i4h_surgical):
        assert Path(module.__file__).is_relative_to(numerical)
    parity = read("exported/parity.json")
    learning = read("learning_checks.json")
    baseline = json.loads(args.baseline.read_text()) if args.baseline else read("bc_20.json")
    starts = folder / "validation_20.json"
    assert baseline["episodes"] == 20 and baseline["objective"] == args.objective
    assert baseline["starts_sha256"] == digest(starts)
    assert learning["checkpoint_sha256"] == digest(folder / "bc_best.pt")
    assert parity["max_analytic_kl"] == 0 and parity["mean_max_abs_error"] == 0
    for worlds in (1, 8, 16):
        grasped = read(f"grasped_isolation_{worlds}.json")
        assert all(
            grasped[key]
            for key in (
                "active_grasps",
                "masked_restore_isolation",
                "ordinary_reset_isolation",
                "exact_complete_round_trip",
                "incompatible_bundle_rejected",
                "training_recorders_disabled",
            )
        )
        assert min(grasped["original_patch_sizes"]) > 0
        isolation = read(f"isolation_{worlds}.json")
        replay = read(f"fresh_replay_{worlds}.json")
        assert (
            isolation["worlds"] == worlds
            and isolation["masked_restore_isolation"]
            and isolation["ordinary_reset_isolation"]
        )
        assert isolation["observed_physics_steps"] == 2 and isolation["pure_getters"]
        assert isolation["max_action_displacement_m"] <= 0.00100001
        assert (
            replay["initial_observation_exact"] and replay["fresh_process"] and replay["max_particle_error_m"] <= 1e-5
        )
    tests = args.tests.read_text()
    assert re.search(r"\d+ passed", tests) and "failed" not in tests
    assert all(v["passed"] for v in learning["checks"].values())
    packages = {
        name: metadata.version(name)
        for name in ("isaacsim", "isaaclab", "isaaclab-arena", "newton", "warp-lang", "rsl-rl-lib", "torch")
    }
    sources = [
        root / "arena/i4h_arena/medical/liver.py",
        root / "arena/i4h_arena/medical/liver_controller.py",
        root / "arena/i4h_arena/scenes/liver_reach_retract.py",
        root / "arena/i4h_arena/envcfg/liver_reach_retract/calibration.json",
        root / "rl/i4h_rl/liver_ppo.py",
        *sorted((root / "arena/i4h_arena/envcfg/liver_reach_retract").glob("*.py")),
    ]
    manifest = dict(
        objective=args.objective,
        gates={name: True for name in REQUIRED_GATES},
        numerical_revision=revision,
        workflows_baseline="1208154b83180901aedcf8eb23371c9045ea3d88",
        versions=packages,
        bc_checkpoint_sha256=digest(folder / "bc_best.pt"),
        bc_successes=baseline["successes"],
        validation_starts=str(starts),
        validation_starts_sha256=digest(starts),
        tests_sha256=digest(args.tests),
        qualified_source_hashes={str(p.relative_to(root)): digest(p) for p in sources},
    )
    write_json(args.output, manifest)


if __name__ == "__main__":
    main()

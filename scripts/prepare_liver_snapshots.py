#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Generate strict hosted settled snapshots offline, never from a reset callback."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--worlds", type=int, choices=(1, 8, 16), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--headless", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ["I4H_LIVER_PREPARE"] = "1"
    os.environ["I4H_LIVER_SNAPSHOTS"] = str(args.output.resolve())
    from i4h_arena.app import launch_app
    from i4h_arena.cli import build_parser
    from i4h_arena.scenes.base import load_scene

    options = ["--workflow", "liver_reach_retract", "--mode", "idle", "--seed", "7341", "--envs", str(args.worlds)]
    if args.headless:
        options += ["--headless"]
    cli = build_parser().parse_args(options)
    scene = load_scene("liver_reach_retract", cli, register_assets=False)
    scene.configure_args(cli)
    with launch_app(cli) as app:
        import torch
        from i4h_surgical.grasp_snapshot import HarnessSnapshot

        env = app.make_env(scene)
        with torch.inference_mode():
            env.reset()
            runtime = env.liver
            runtime.settling_previous.assign(runtime.q)
            runtime.settling_state.zero_()
            steps = 0
            streak = np.zeros(args.worlds, int)
            while steps < 1800:
                runtime.held.copy_(runtime.physical)
                env.step(torch.tensor([[0.0, 0.0, 0.0, -1.0]] * args.worlds, device=env.device))
                measurements = runtime.settling_state.numpy()
                steps = int(measurements[:, 0].min())
                streak = measurements[:, 1]
                if steps >= 180 and np.all(streak >= 60):
                    break
            if not np.all(streak >= 60):
                raise RuntimeError("canonical hosted settling failed")
            snapshot = HarnessSnapshot(
                env.i4h.physics, runtime.configuration_identity(), dict(settling_steps=steps, streak=streak.tolist())
            )
            path = args.output / f"settled_{args.worlds}.npz"
            if path.exists():
                raise FileExistsError(f"preserve existing snapshot: {path}")
            snapshot.write(path)
            (args.output / f"manifest_{args.worlds}.json").write_text(json.dumps(snapshot.metadata(), indent=2) + "\n")
            print(f"LIVER_SETTLED_SNAPSHOT {path} {snapshot.hash}", flush=True)
        env.close()


if __name__ == "__main__":
    main()

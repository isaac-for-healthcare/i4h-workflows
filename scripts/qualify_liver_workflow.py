#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Visible normal-Workflow validation with viewport frames and input/action traces."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--mode", choices=("rule-based", "policy"), default="rule-based")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    from i4h_arena.app import launch_app
    from i4h_arena.cli import build_parser
    from i4h_arena.scenes.base import load_scene
    from i4h_engine.loader import apply_overrides, resolve_workflow
    from i4h_rl.liver_experiment import write_json

    cli = build_parser().parse_args(["--workflow", "liver_reach_retract", "--mode", args.mode, "--seed", "7341"])
    scene = load_scene("liver_reach_retract", cli, register_assets=False)
    scene.configure_args(cli)
    with launch_app(cli) as app:
        import torch
        from omni.kit.viewport.utility import get_active_viewport
        from omni.kit.widget.viewport.capture import FileCapture

        from i4h_arena.recording.hdf5 import EpisodeRecorder
        from i4h_arena.runner import SimulationRunner, _render_scene_only

        env = app.make_env(scene)
        workflow = resolve_workflow("liver_reach_retract", args.mode)
        if args.checkpoint:
            apply_overrides(workflow.graph, checkpoint=str(args.checkpoint.resolve()), model_device="cuda:0")
        recorder = EpisodeRecorder(args.output / "workflow.hdf5", workflow=workflow, cameras=())
        original = env.step
        observations, actions, positions, terminals = [], [], [], []

        def step_and_capture(action):
            observations.append(env.liver.observation().cpu().numpy()[0].copy())
            actions.append(action.cpu().numpy()[0].copy())
            result = original(action)
            positions.append(env.liver.q.numpy()[0].copy())
            terminals.extend(result[-1].get("terminal_results", []))
            if len(actions) in (10, 40, 80, 120):
                path = (args.output / f"viewport_{len(actions):04d}.png").resolve()
                for _ in range(3):
                    _render_scene_only(env, app.app)
                get_active_viewport().schedule_capture(FileCapture(str(path)))
                for _ in range(5):
                    _render_scene_only(env, app.app)
                if not path.is_file() or path.stat().st_size == 0:
                    raise RuntimeError("viewport frame was not captured")
            return result

        env.step = step_and_capture
        try:
            with torch.no_grad():
                summary = SimulationRunner(
                    scene=scene,
                    workflow=workflow,
                    env=env,
                    app=app.app,
                    recorder=recorder,
                    seed=7341,
                    episodes=1,
                    attempts=1,
                    record_failures=True,
                ).run()
            write_json(
                args.output / "result.json",
                dict(
                    mode=args.mode,
                    successes=summary.succeeded,
                    episodes=1,
                    steps=[episode.steps for episode in summary.episodes],
                    terminal_results=terminals,
                ),
            )
            np.savez_compressed(
                args.output / "trace.npz", observations=observations, actions=actions, positions=positions
            )
            print(summary.render(), flush=True)
        finally:
            recorder.close()
            env.close()


if __name__ == "__main__":
    main()

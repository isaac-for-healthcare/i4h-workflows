#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Measure matched navigation phases without imposing a new performance gate."""

import argparse
import faulthandler
import json
import os
import time
from contextlib import suppress
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--backend", choices=("numerical", "legacy"), required=True)
    parser.add_argument("--workflow", default="endoluminal_navigation")
    parser.add_argument("--worlds", type=int, choices=(1, 16, 64), default=1)
    parser.add_argument("--patient-twin")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    os.environ["I4H_WORKFLOWS"] = str(Path(__file__).resolve().parents[1])
    from qualify_navigation_native import source_revisions

    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    cli = argparse.Namespace(
        num_envs=args.worlds,
        seed=42,
        headless=not args.visible,
        visualizer="kit" if args.visible else "none",
        device="cuda:0",
        enable_cameras=True,
        presets="newton",
        no_cameras=False,
        solve_relations=False,
        catheter_backend=args.backend,
        patient_twin=args.patient_twin,
        fluoro_backend=None,
        episode_steps=600,
    )
    app = AppLauncher(cli).app
    faulthandler.dump_traceback_later(90, repeat=True)
    import gymnasium as gym
    import torch
    import warp as wp

    from i4h_arena.medical.fluoroscopy_sources import bind_fluoroscopy_sources
    from i4h_arena.scenes.base import load_scene

    totals = dict(physics_s=0.0, viewport_render_s=0.0, fluoroscopy_render_s=0.0, input_s=0.0, control_loop_s=0.0)
    counts = dict.fromkeys(totals, 0)
    scene = load_scene(args.workflow, cli, register_assets=False)

    def configure(cfg):
        cfg.recorders = None
        return cfg

    try:
        gym_id, cfg = scene.gym_spec(config_transform=configure)
        env = gym.make(gym_id, cfg=cfg).unwrapped
        try:
            env.reset(seed=42)
            bind_fluoroscopy_sources(env)
            sensor = env.scene["fluoroscopy"]
            action = torch.tensor([[0.009, 0.1, 0.1, 0.1]], device=env.device).repeat(args.worlds, 1)
            print("profile: warming up 5 control steps", flush=True)
            for _ in range(5):
                env.step(action)
                _ = sensor.data
            keyboard = None
            if args.visible:
                from i4h_arena.devices.catheter import CatheterKeyboard, CatheterKeyboardCfg

                keyboard = CatheterKeyboard(CatheterKeyboardCfg(sim_device=env.device))

            def timed(owner, name, key):
                original = getattr(owner, name)

                def call(*args, **kwargs):
                    wp.synchronize()
                    start = time.perf_counter()
                    result = original(*args, **kwargs)
                    wp.synchronize()
                    totals[key] += time.perf_counter() - start
                    counts[key] += 1
                    return result

                setattr(owner, name, call)

            timed(env.sim, "step", "physics_s")
            timed(env.sim, "render", "viewport_render_s")
            timed(sensor, "_update_buffers_impl", "fluoroscopy_render_s")
            print("profile: measuring control steps", flush=True)
            with torch.inference_mode():
                for index in range(args.steps):
                    wp.synchronize()
                    start = time.perf_counter()
                    input_start = time.perf_counter()
                    if keyboard is not None:
                        keyboard.advance()
                    else:
                        action.clone()
                    totals["input_s"] += time.perf_counter() - input_start
                    counts["input_s"] += 1
                    env.step(action)
                    _ = sensor.data
                    wp.synchronize()
                    totals["control_loop_s"] += time.perf_counter() - start
                    counts["control_loop_s"] += 1
                    if (index + 1) % 5 == 0:
                        print(f"profile: {index + 1}/{args.steps} control steps", flush=True)
            if args.visible:
                assert counts["viewport_render_s"] > 0, "visible profile did not render a viewport"
            result = dict(
                visible=args.visible,
                active_visualizers=[type(item).__name__ for item in env.sim.visualizers],
                backend=args.backend,
                worlds=args.worlds,
                workflow=args.workflow,
                patient=args.patient_twin,
                control_hz=30,
                physics_hz=120,
                decimation=4,
                totals=totals,
                counts=counts,
                mean_ms={key: 1000 * value / max(counts[key], 1) for key, value in totals.items()},
                input_measurement="keyboard polling" if keyboard else "scripted tensor clone",
                acceptance_threshold=None,
                **source_revisions(__file__),
            )
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            del keyboard
        finally:
            env.close()
    finally:
        faulthandler.cancel_dump_traceback_later()
        with suppress(SystemExit):
            app.close()


if __name__ == "__main__":
    main()

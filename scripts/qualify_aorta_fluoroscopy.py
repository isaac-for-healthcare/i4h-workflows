#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Capture independent aorta demos, freeze calibration, then check a launcher replay."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

WORKFLOW = "endoluminal_aorta_fluoroscopy"


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def sample(env):
    from i4h_arena.scenes.endoluminal_aorta_fluoroscopy import AortaStateProvider

    provider = AortaStateProvider(env)
    points, targets = provider.points()[0], provider.targets()[0]
    velocities = env.i4h.scene.qd.numpy().copy()
    sensor = env.scene["fluoroscopy"]
    sensor.update(0.0, force_recompute=True)
    guidance = sensor.data.output["guidance"][0].detach().cpu().numpy().copy()
    for value in (points, targets, velocities):
        assert np.isfinite(value).all(), "Non-finite physical state"
    return {"points": points, "targets": targets, "velocities": velocities, "guidance": guidance}


def demo(run_dir: Path, eager: bool):
    from contextlib import closing

    from isaaclab.app import AppLauncher

    run_dir.mkdir(parents=True, exist_ok=True)
    args = argparse.Namespace(
        num_envs=1,
        seed=0,
        headless=False,
        device="cuda:0",
        enable_cameras=True,
        presets="newton",
        no_cameras=False,
        solve_relations=False,
    )
    with closing(AppLauncher(args).app):
        import gymnasium as gym
        import h5py
        import torch
        from isaaclab_arena.policy.policy_base import PolicyCfg
        from isaaclab_newton.physics import NewtonManager
        from PIL import Image

        from i4h_arena.policies.aorta import AortaFluoroscopySweepPolicy
        from i4h_arena.scenes.base import load_scene

        scene = load_scene(WORKFLOW, args, register_assets=False)

        def configure(cfg):
            cfg.recorders = None
            cfg.sim.physics.use_cuda_graph = not eager
            return cfg

        gym_id, cfg = scene.gym_spec(config_transform=configure)
        env = gym.make(gym_id, cfg=cfg).unwrapped
        try:
            env.reset(seed=0)
            pristine = sample(env)
            policy = AortaFluoroscopySweepPolicy(PolicyCfg())
            samples, actions = [], []
            with torch.inference_mode():
                for _ in range(115):
                    action = policy.get_action(env, None)
                    actions.append(action.cpu().numpy()[0].copy())
                    env.step(action)
                    samples.append(sample(env))
                captured = NewtonManager._graph is not None
                assert captured == (not eager)
                env.reset(seed=0)
                after_reset = sample(env)
                env.step(torch.zeros((1, 3), device=env.device))
                after_reset_step = sample(env)
            arrays = {name: np.stack([value[name] for value in samples]) for name in samples[0]}
            arrays.update(actions=np.asarray(actions), captured=np.asarray(captured))
            for prefix, state in (("pristine", pristine), ("reset", after_reset), ("reset_step", after_reset_step)):
                arrays.update({f"{prefix}_{name}": value for name, value in state.items()})
            np.savez_compressed(run_dir / "samples.npz", **arrays)
            for name, frame in (
                ("start", pristine["guidance"]),
                ("insertion-end", samples[44]["guidance"]),
                ("reset", after_reset["guidance"]),
            ):
                Image.fromarray(frame).save(run_dir / f"{name}.png")
            # Diagnostic archive for the existing offline calibration/comparison tools.
            with h5py.File(run_dir / "demo.hdf5", "w") as output:
                data = output.create_group("data")
                data.attrs["env_args"] = json.dumps(dict(env_name=WORKFLOW))
                episode = data.create_group("demo_0")
                episode.attrs["success"] = True
                episode.attrs["diagnostic_only"] = True
                episode.create_group("initial_state")
                episode.create_dataset("actions", data=arrays["actions"])
                episode.create_dataset("obs/fluoroscopy", data=arrays["guidance"])
                episode.create_dataset("diagnostics/tip_world_m", data=arrays["points"][:, -1])
            (run_dir / "result.json").write_text(
                json.dumps(
                    dict(
                        diagnostic_only=True,
                        samples=115,
                        seed=0,
                        captured=captured,
                        physics_hz=60,
                        decimation=2,
                        samples_sha256=digest(run_dir / "samples.npz"),
                    ),
                    indent=2,
                )
            )
        finally:
            env.close()


def spread(a, b):
    return float(np.linalg.norm(np.asarray(a, dtype=np.float64) - b, axis=-1).max())


def pixels(points, frames):
    # Independent acceptance projection; do not call the renderer's projection.
    tip = points[:, -1]
    expected = np.stack((255 * (tip[:, 0] + 0.5) / 11.5, 255 * (2.0 - tip[:, 1]) / 4.0), axis=-1)
    assert ((expected >= 0) & (expected <= 255)).all(), "Tip projects outside the detector"
    errors = []
    for projected, frame in zip(expected, frames, strict=True):
        y, x = np.nonzero(np.all(frame == (245, 255, 245), axis=-1))
        assert len(x), "No rendered distal marker centre"
        centre = np.asarray(((x.min() + x.max()) / 2, (y.min() + y.max()) / 2))
        errors.append(float(np.linalg.norm(centre - projected)))
    assert max(errors) <= 2.0, f"Marker projection error: {max(errors)} px"
    return max(errors)


def calibrate(first: Path, second: Path, output: Path):
    with np.load(first / "samples.npz") as a, np.load(second / "samples.npz") as b:
        assert a["captured"] and b["captured"], "Eager diagnostic runs cannot calibrate the captured stack"
        s_reset = max(spread(a["pristine_points"], b["pristine_points"]), spread(a["points"][0], b["points"][0]))
        s_tip = spread(a["points"][:, -1], b["points"][:, -1])
        t_reset, t_replay = max(1e-5, 3 * s_reset), max(1e-5, 3 * s_tip)
        assert t_reset <= 1e-4, f"Reset calibration exceeds cap: {t_reset} m"
        assert t_replay <= 1e-3, f"Replay calibration exceeds cap: {t_replay} m"
        runs = []
        for data in (a, b):
            assert data["actions"].shape == (115, 3)
            for name in data.files:
                assert np.isfinite(data[name]).all(), f"Non-finite {name}"
            insertion = float(data["targets"][44, 0] - data["targets"][14, 0])
            motion = float(np.linalg.norm(data["points"][44, -1] - data["points"][14, -1]))
            assert abs(insertion - 0.5) <= 0.001, f"Insertion advance: {insertion} m"
            assert motion >= 0.05, f"Tip motion: {motion} m"
            reset_error = spread(data["pristine_points"], data["reset_points"])
            reset_step_error = spread(data["points"][0], data["reset_step_points"])
            assert max(reset_error, reset_step_error) <= t_reset, (reset_error, reset_step_error, t_reset)
            np.testing.assert_array_equal(data["pristine_targets"], np.zeros(3))
            np.testing.assert_array_equal(data["reset_targets"], data["pristine_targets"])
            np.testing.assert_array_equal(data["reset_step_targets"], data["targets"][0])
            runs.append(
                {
                    "insertion_advance_m": insertion,
                    "tip_motion_m": motion,
                    "reset_error_m": reset_error,
                    "reset_step_error_m": reset_step_error,
                    "max_marker_error_px": pixels(data["points"], data["guidance"]),
                }
            )
        np.testing.assert_array_equal(a["actions"], b["actions"])
    result = {
        "S_reset_m": s_reset,
        "T_reset_m": t_reset,
        "S_tip_m": s_tip,
        "T_replay_m": t_replay,
        "runs": runs,
        "first": str(first),
        "second": str(second),
        "first_sha256": digest(first / "samples.npz"),
        "second_sha256": digest(second / "samples.npz"),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def replay_check(calibration: Path, source: Path, replay: Path, output: Path):
    import h5py
    from PIL import Image

    thresholds = json.loads(calibration.read_text())
    with h5py.File(source) as original, h5py.File(replay) as played:
        a, b = original["data/demo_0"], played["data/demo_0"]
        assert a.attrs["success"] and b.attrs["success"]
        assert a["actions"].shape == b["actions"].shape == (115, 3)
        np.testing.assert_array_equal(a["actions"][()], b["actions"][()])
        frames = b["obs/fluoroscopy"][()]
        assert frames.shape == (115, 256, 256, 3)
        for frame in frames:
            image = Image.fromarray(frame)
            assert image.size == (256, 256) and image.mode == "RGB"
        errors = np.linalg.norm(a["diagnostics/tip_world_m"][()] - b["diagnostics/tip_world_m"][()], axis=-1)
        assert np.isfinite(errors).all() and float(errors.max()) <= thresholds["T_replay_m"]
        for key in b["diagnostics"]:
            assert np.isfinite(b["diagnostics"][key][()]).all()
    result = {
        "success": True,
        "samples": 115,
        "actions_identical_including_final": True,
        "decoded_frames": 115,
        "max_tip_error_m": float(errors.max()),
        "T_replay_m": thresholds["T_replay_m"],
        "calibration_sha256": digest(calibration),
        "source_sha256": digest(source),
        "replay_sha256": digest(replay),
    }
    output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="operation", required=True)
    capture = commands.add_parser("demo")
    capture.add_argument("--run-dir", required=True, type=Path)
    capture.add_argument("--eager", action="store_true", help="diagnostic only; never calibrates captured acceptance")
    calibration = commands.add_parser("calibrate")
    calibration.add_argument("first", type=Path)
    calibration.add_argument("second", type=Path)
    calibration.add_argument("--output", required=True, type=Path)
    replay = commands.add_parser("replay-check")
    for name in ("calibration", "source", "replay", "output"):
        replay.add_argument(f"--{name}", required=True, type=Path)
    args = parser.parse_args()
    if args.operation == "demo":
        demo(args.run_dir.resolve(), args.eager)
    elif args.operation == "calibrate":
        calibrate(args.first.resolve(), args.second.resolve(), args.output.resolve())
    else:
        replay_check(args.calibration, args.source, args.replay, args.output)


if __name__ == "__main__":
    main()

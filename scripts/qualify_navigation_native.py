#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Native navigation lifecycle and complete continuation checks, one simulator at a time."""

from __future__ import annotations

import argparse
import json
import os
from argparse import Namespace
from contextlib import suppress
from pathlib import Path

import numpy as np


def equal(left, right, prefix=""):
    assert left.keys() == right.keys(), prefix
    for key, value in left.items():
        name = f"{prefix}/{key}"
        if isinstance(value, dict):
            equal(value, right[key], name)
        else:
            assert value.shape == right[key].shape and value.dtype == right[key].dtype, name
            assert value.tobytes() == right[key].tobytes(), name


def finalize_result(result):
    """Exact snapshots alone do not qualify subsequent continuation trajectories."""
    failures = [
        name
        for name in ("same_process_trajectory_exact", "fresh_trajectory_exact")
        if name in result and not result[name]
    ]
    failures.extend(
        name
        for name in ("same_process_robot_max_abs_rad", "fresh_robot_max_abs_rad")
        if name in result and result[name] != 0.0
    )
    if failures:
        result.update(passed=False, continuation_failures=failures)
    return result


def close_qualification(app, result):
    """Isaac Sim fast shutdown exits inside close, so pass its status explicitly."""
    exit_code = 0 if result.get("passed", False) else 1
    with suppress(SystemExit):
        app.close(exit_code=exit_code)
    if exit_code:
        raise SystemExit(exit_code)


def source_revisions(driver_path):
    import hashlib
    import subprocess

    root = Path(__file__).resolve().parents[1]
    import i4h_physics

    numerical = Path(i4h_physics.__file__).resolve().parent
    return dict(
        workflow_revision=subprocess.check_output(["git", "-C", str(root), "rev-parse", "HEAD"], text=True).strip(),
        numerical_revision=subprocess.check_output(
            ["git", "-C", str(numerical), "rev-parse", "HEAD"], text=True
        ).strip(),
        tracked_changes=subprocess.check_output(
            ["git", "-C", str(root), "diff", "--name-only", "HEAD"], text=True
        ).splitlines(),
        driver_sha256=hashlib.sha256(Path(driver_path).read_bytes()).hexdigest(),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument("--workflow", default="endoluminal_navigation")
    parser.add_argument("--worlds", type=int, choices=(1, 8, 16), default=1)
    parser.add_argument("--patient-twin")
    parser.add_argument("--visible", action="store_true")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--reconstruct", action="store_true")
    parser.add_argument("--warmup", type=int, default=6)
    parser.add_argument("--output", required=True, type=Path)
    args = parser.parse_args()
    os.environ["I4H_WORKFLOWS"] = str(Path(__file__).resolve().parents[1])
    os.environ.setdefault("OMNI_KIT_ACCEPT_EULA", "YES")
    from isaaclab.app import AppLauncher

    cli = Namespace(
        num_envs=args.worlds,
        seed=42,
        headless=not args.visible,
        visualizer="kit" if args.visible else "none",
        device="cuda:0",
        enable_cameras=True,
        presets="newton",
        no_cameras=False,
        solve_relations=False,
        catheter_backend="numerical",
        patient_twin=args.patient_twin,
        fluoro_backend=None,
        episode_steps=600,
    )
    app = AppLauncher(cli).app
    import gymnasium as gym
    import torch

    from i4h_arena.scenes.base import load_scene

    result = dict(
        workflow=args.workflow,
        worlds=args.worlds,
        patient=args.patient_twin,
        visible=args.visible,
        warmup_control_steps=0 if args.resume else args.warmup,
        **source_revisions(__file__),
    )
    try:
        scene = load_scene(args.workflow, cli, register_assets=False)

        def configure(cfg):
            cfg.recorders = None
            return cfg

        gym_id, cfg = scene.gym_spec(config_transform=configure)
        env = gym.make(gym_id, cfg=cfg).unwrapped
        try:
            with torch.inference_mode():
                env.reset(seed=42)
                result["active_visualizers"] = [type(item).__name__ for item in env.sim.visualizers]
                if args.visible:
                    assert (
                        "KitVisualizer" in result["active_visualizers"]
                    ), "visible check requires an active Kit viewport"
                action = torch.tensor([0.009, 0.2, 0.15, 0.12], device=env.device).repeat(args.worlds, 1)

                def observation_arrays(value):
                    if isinstance(value, dict):
                        return {key: observation_arrays(item) for key, item in value.items()}
                    return value.detach().cpu().numpy().copy()

                if args.resume:
                    archive = np.load(args.resume)
                    bundle = {
                        family: {
                            key.split("__", 1)[1]: archive[key]
                            for key in archive.files
                            if key.startswith(family + "__")
                        }
                        for family in ("numerical", "host")
                    }
                    bundle["identity"] = archive["identity"]
                    env.restore_complete_state(bundle)
                    equal(bundle, env.capture_complete_state())
                    result["fresh_state_roundtrip"] = True
                    expected = {}
                    for key in archive.files:
                        if key.startswith("observation__"):
                            parts = key.split("__")[1:]
                            target = expected
                            for part in parts[:-1]:
                                target = target.setdefault(part, {})
                            target[parts[-1]] = archive[key]
                    if expected:
                        equal(expected, observation_arrays(env.observation_manager.compute(update_history=False)))
                        result["fresh_observation_roundtrip"] = True
                    errors = []
                    robot_errors = []
                    for index, expected in enumerate(archive["trajectory"]):
                        env.step(action)
                        actual = env.scene["catheter"].data.positions_world_m.cpu().numpy().copy()
                        errors.append(float(np.max(np.abs(actual - expected))))
                        if "robot_trajectory" in archive:
                            joints = env.scene["robot"].data.joint_pos.torch.cpu().numpy()
                            robot_errors.append(float(np.max(np.abs(joints - archive["robot_trajectory"][index]))))
                    if robot_errors:
                        result["fresh_robot_max_abs_rad"] = max(robot_errors)
                    result.update(
                        fresh_trajectory_exact=max(errors) == 0, fresh_trajectory_max_abs_m=max(errors), passed=True
                    )
                    return
                for _ in range(args.warmup):
                    env.step(action)

                drive = env.action_manager.get_term("catheter")
                if hasattr(drive, "_approach"):
                    result["approach_parked_after_warmup"] = bool(drive._approach.parked.all())
                expected_observation = observation_arrays(env.observation_manager.compute(update_history=False))
                before = env.capture_complete_state()
                saved = env.capture_complete_state([args.worlds // 2])
                sensor_cfg = env.scene["fluoroscopy"].cfg
                gamma = sensor_cfg.dsa_gamma
                sensor_cfg.dsa_gamma = gamma + 0.1
                try:
                    env.restore_complete_state(before)
                except ValueError as error:
                    assert "configuration" in str(error)
                    result["configuration_mismatch_rejected"] = True
                else:
                    raise AssertionError("changed fluoroscopy configuration was accepted")
                finally:
                    sensor_cfg.dsa_gamma = gamma
                equal(before, env.capture_complete_state())
                trajectory = []
                robot_trajectory = []
                for _ in range(4):
                    env.step(action)
                    trajectory.append(env.scene["catheter"].data.positions_world_m.cpu().numpy().copy())
                    if "robot" in env.scene.articulations:
                        robot_trajectory.append(env.scene["robot"].data.joint_pos.torch.cpu().numpy().copy())
                env.restore_complete_state(before)
                equal(before, env.capture_complete_state())
                result["exact_state_roundtrip"] = True
                restored_observation = observation_arrays(env.observation_manager.compute(update_history=False))
                equal(expected_observation, restored_observation)
                result["exact_observation_roundtrip"] = True
                trajectory_errors = []
                robot_errors = []
                for index, expected in enumerate(trajectory):
                    env.step(action)
                    actual = env.scene["catheter"].data.positions_world_m.cpu().numpy().copy()
                    assert np.isfinite(actual).all()
                    trajectory_errors.append(float(np.max(np.abs(actual - expected))))
                    if robot_trajectory:
                        joints = env.scene["robot"].data.joint_pos.torch.cpu().numpy()
                        robot_errors.append(float(np.max(np.abs(joints - robot_trajectory[index]))))
                result["same_process_trajectory_exact"] = max(trajectory_errors) == 0.0
                result["same_process_trajectory_max_abs_m"] = max(trajectory_errors)
                if robot_errors:
                    result["same_process_robot_max_abs_rad"] = max(robot_errors)
                ids = torch.tensor([args.worlds // 2], device=env.device)
                peer_before = env.capture_complete_state()
                env._reset_idx(ids)
                peer_after = env.capture_complete_state()
                for family in ("numerical", "host"):
                    for key, value in peer_before[family].items():
                        peers = np.arange(args.worlds) != args.worlds // 2
                        assert value[peers].tobytes() == peer_after[family][key][peers].tobytes(), f"reset peer {key}"
                env.restore_complete_state(saved, ids)
                equal(saved, env.capture_complete_state(ids))
                result["selected_reset_restore"] = True
                limit_state = env.capture_complete_state()
                drive = env.action_manager.get_term("catheter")
                limit = float(drive.cfg.max_tip_bend_rad)
                for sign in (1.0, -1.0):
                    drive._tip_bend_angle.fill_(sign * (limit - 1e-4))
                    env.step(torch.full_like(action, sign * 1000.0))
                    assert torch.all(drive._processed_actions[:, 0].abs() <= drive.cfg.max_insertion_velocity_mps)
                    assert torch.allclose(drive._tip_bend_angle, torch.full_like(drive._tip_bend_angle, sign * limit))
                result["native_rate_and_bend_limits"] = True
                env.restore_complete_state(limit_state)
                pre_terminal = env.capture_complete_state()
                env.episode_length_buf[ids] = env.max_episode_length - 1
                # Force a timeout fixture without violating the tracker's
                # consecutive-step invariant for patient tasks.
                if env.progress_tracker is not None:
                    env.progress_tracker._last_processed_step[ids] = env.max_episode_length - 1
                env.step(action)
                terminal = env.extras["navigation_terminal"]
                assert torch.any(terminal["env_ids"] == ids[0])
                assert torch.all(terminal["episode_length"] == env.max_episode_length)
                assert env.episode_length_buf[ids[0]] == 0
                assert not torch.equal(terminal["points"][0], env.scene["catheter"].data.positions_world_m[ids[0]])
                result["terminal_before_reset"] = True
                env.restore_complete_state(pre_terminal)

                if args.reconstruct:
                    complete = env.capture_complete_state()
                    env.sim.reset(soft=False)
                    env.restore_complete_state(complete)
                    equal(complete, env.capture_complete_state())
                    env.step(action)
                    result["reconstruction_roundtrip"] = True
                from PIL import Image

                from i4h_arena.medical.fluoroscopy_sources import bind_fluoroscopy_sources

                bind_fluoroscopy_sources(env)
                sensor = env.scene["fluoroscopy"]
                sensor.update(0.0, force_recompute=True)
                Image.fromarray(sensor.data.output["rgb"][0].cpu().numpy()).save(args.output.with_suffix(".png"))
                if args.visible:
                    env.sim.set_camera_view(eye=(2.45, -1.65, 1.65), target=(-0.25, 0.12, 0.78))
                    import carb

                    from i4h_arena.devices.catheter import CatheterKeyboard, CatheterKeyboardCfg

                    keyboard = CatheterKeyboard(CatheterKeyboardCfg(sim_device=env.device))
                    # Exercise Kit's event fallback explicitly; physical key polling
                    # requires an operator and is reported separately.
                    keyboard._buttons.clear()
                    for key, channel in (("W", 0), ("D", 1), ("C", 2), ("Q", 3)):
                        event = Namespace(
                            input=getattr(carb.input.KeyboardInput, key), type=carb.input.KeyboardEventType.KEY_PRESS
                        )
                        keyboard._on_keyboard_event(event)
                        command = keyboard.advance()
                        assert command[channel] > 0
                        env.step(command.repeat(args.worlds, 1))
                        event.type = carb.input.KeyboardEventType.KEY_RELEASE
                        keyboard._on_keyboard_event(event)
                        assert not torch.any(keyboard.advance())
                    result["keyboard_event_path"] = True
                    result["keyboard_hardware_input_tested"] = False
                    del keyboard
                    from omni.kit.viewport.utility import get_active_viewport
                    from omni.kit.widget.viewport.capture import FileCapture

                    for _ in range(8):
                        app.update()
                    # Give evidence capture an explicit camera. The active Kit
                    # perspective camera can retain an unrelated default pose
                    # after the host reconstructs its Fabric bindings.
                    from pxr import Gf, UsdGeom

                    camera = UsdGeom.Camera.Define(env.sim.stage, "/QualificationCamera")
                    camera.GetFocalLengthAttr().Set(35.0)
                    transform = UsdGeom.Xformable(camera.GetPrim())
                    transform.ClearXformOpOrder()
                    transform.AddTransformOp().Set(
                        Gf.Matrix4d()
                        .SetLookAt(Gf.Vec3d(2.45, -1.65, 1.65), Gf.Vec3d(-0.25, 0.12, 0.78), Gf.Vec3d(0, 0, 1))
                        .GetInverse()
                    )
                    viewport = get_active_viewport()
                    viewport.camera_path = camera.GetPath().pathString
                    for _ in range(8):
                        app.update()
                    viewport.schedule_capture(FileCapture(str(args.output.with_suffix(".viewport.png"))))
                    for _ in range(8):
                        app.update()
                # A one-episode snapshot is serialized independently of batch size.
                flat = {
                    f"{family}__{key}": value
                    for family in ("numerical", "host")
                    for key, value in saved[family].items()
                }
                flat["identity"] = saved["identity"]
                selected = args.worlds // 2

                def add_observations(value, prefix="observation"):
                    for key, item in value.items():
                        name = f"{prefix}__{key}"
                        if isinstance(item, dict):
                            add_observations(item, name)
                        else:
                            flat[name] = item[selected : selected + 1]

                add_observations(expected_observation)
                flat["trajectory"] = np.stack([value[selected : selected + 1] for value in trajectory])
                if robot_trajectory:
                    flat["robot_trajectory"] = np.stack([value[selected : selected + 1] for value in robot_trajectory])
                np.savez_compressed(args.output.with_suffix(".npz"), **flat)
                result["passed"] = True
        finally:
            env.close()
    except BaseException as error:
        import traceback

        traceback.print_exc()
        result.update(passed=False, error=repr(error))
    finally:
        finalize_result(result)
        args.output.write_text(json.dumps(result, indent=2) + "\n")
        # Keep this inside finally: the fresh-process branch returns from try.
        close_qualification(app, result)


if __name__ == "__main__":
    main()

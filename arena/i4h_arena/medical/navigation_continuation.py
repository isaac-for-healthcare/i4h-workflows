# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Portable, selected-episode continuation for native navigation."""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from pathlib import Path

import numpy as np
import torch
import warp as wp
from isaaclab.managers import RecorderTerm

from i4h_common.catheter_backend import CONTINUATION_SCHEMA


def tensor_bundle(bundle):
    return {
        key: tensor_bundle(value) if isinstance(value, dict) else torch.from_numpy(value.copy())
        for key, value in bundle.items()
    }


def numpy_bundle(bundle):
    return {
        key: numpy_bundle(value) if isinstance(value, dict) else value.detach().cpu().numpy().copy()
        for key, value in bundle.items()
    }


class NavigationInitialStateRecorder(RecorderTerm):
    def record_post_reset(self, env_ids):
        return "initial_state/navigation", tensor_bundle(self._env.capture_complete_state(env_ids))


def configuration_identity(env):
    if hasattr(env, "_navigation_identity"):
        return env._navigation_identity
    spec = asdict(env.cfg.navigation_rod_spec)
    spec.pop("num_envs", None)
    spec.pop("device", None)
    digest = hashlib.sha256(json.dumps(spec, sort_keys=True).encode())
    from importlib import import_module

    for name in ("i4h_physics", "i4h_endoluminal", "i4h_isaaclab"):
        root = Path(import_module(name).__file__).parent
        for file in sorted(root.rglob("*.py")):
            digest.update(str(file.relative_to(root)).encode())
            digest.update(file.read_bytes())
    arena = Path(__file__).resolve().parents[1]
    for name in (
        "medical/catheter_drive.py",
        "medical/numerical_catheter.py",
        "embodiments/catheter.py",
        "embodiments/franka_catheter.py",
        "envcfg/endoluminal_navigation/numerical.py",
        "medical/navigation_continuation.py",
        "medical/navigation_reward.py",
        "medical/navigation_goal.py",
    ):
        digest.update(name.encode())
        digest.update((arena / name).read_bytes())
    if spec["patient_twin_manifest"]:
        from .patient_twin import PatientTwin

        manifest = Path(spec["patient_twin_manifest"])
        digest.update(manifest.read_bytes())
        for file in PatientTwin.load(manifest).artifacts.values():
            if isinstance(file, Path) and file.is_file():
                digest.update(file.read_bytes())
    env._navigation_identity = np.frombuffer(
        f"numerical:{CONTINUATION_SCHEMA}:{digest.hexdigest()}".encode(), np.uint8
    ).copy()
    return env._navigation_identity


def host_arrays(env):
    from .navigation_goal import HOLD_COUNTER_ATTR, hold_counter
    from .navigation_reward import REMAINING_ARC_ATTR

    catheter = env.scene["catheter"]
    drive = env.action_manager.get_term("catheter")
    carm = env.action_manager.get_term("carm_orbit")
    result = {
        "episode_length": env.episode_length_buf,
        "action": env.action_manager.action,
        "previous_action": env.action_manager.prev_action,
        "arrival_hold": hold_counter(env),
    }
    if not hasattr(env, "reward_buf"):
        env.reward_buf = torch.zeros(env.num_envs, device=env.device)
    if not hasattr(env, "reset_terminated"):
        env.reset_terminated = env.termination_manager.terminated
    if not hasattr(env, "reset_time_outs"):
        env.reset_time_outs = env.termination_manager.time_outs
    for name in ("reward_buf", "reset_buf", "reset_terminated", "reset_time_outs"):
        if hasattr(env, name):
            result[name] = getattr(env, name)
    for name in ("_term_dones", "_last_episode_dones", "_truncated_buf", "_terminated_buf"):
        result[f"termination{name}"] = getattr(env.termination_manager, name)
    # Capture the current tensor: arrival replaces this env attribute each evaluation.
    result["arrival_hold"] = getattr(env, HOLD_COUNTER_ATTR)
    if not hasattr(env, REMAINING_ARC_ATTR):
        setattr(env, REMAINING_ARC_ATTR, torch.full((env.num_envs,), float("nan"), device=env.device))
    result["remaining_arc"] = getattr(env, REMAINING_ARC_ATTR)
    for label, owner, names in (
        ("ik", getattr(drive, "_ik", None), ("ee_pos_des", "ee_quat_des", "_command")),
        ("drive", drive, ("_raw_actions", "_processed_actions", "_tip_bend_angle")),
        ("carm", carm, ("_raw_actions", "_processed_actions", "_angle_rad", "_root_position", "_root_orientation")),
        ("catheter", catheter._data, ("insertion_m", "rotation_rad", "command")),
        ("rail", getattr(drive, "_rail", None), ("_arc_m", "_twist_rad")),
        ("introducer", getattr(drive, "_introducer", None), ("_depth_m", "_prev_flange_pos", "_primed")),
        (
            "approach",
            getattr(drive, "_approach", None),
            (
                "_latched",
                "_elapsed_s",
                "_start_pos",
                "_start_quat",
                "_target_pos",
                "_target_quat",
                "_command_pos",
                "_command_quat",
            ),
        ),
    ):
        if owner is not None:
            for name in names:
                result[f"{label}{name}"] = getattr(owner, name)
    coupled = env.i4h.physics.coupled
    if coupled is not None:
        rigid = coupled.rigid
        for name in ("qacc_warmstart", "qfrc_applied", "xfrc_applied", "ctrl", "act", "time"):
            result[f"mj_{name}"] = wp.to_torch(getattr(rigid.mjw_data, name))
        for family, container, names in (
            ("rigid_state", env.i4h.physics.state, ("body_q", "body_qd", "body_f", "joint_q", "joint_qd")),
            ("rigid_control", env.i4h.physics.control, ("joint_target_pos", "joint_target_vel", "joint_f")),
        ):
            for name in names:
                value = getattr(container, name, None)
                if value is not None:
                    result[f"{family}_{name}"] = wp.to_torch(value).view(env.num_envs, -1, *value.numpy().shape[1:])
        robot = env.scene["robot"]
        result["robot_joint_pos"] = robot.data.joint_pos.torch
        result["robot_joint_vel"] = robot.data.joint_vel.torch
    tracker = env.progress_tracker
    if tracker is not None:
        result["progress_success"] = tracker._task_success
        result["progress_last_step"] = tracker._last_processed_step
        for index, runner in enumerate(tracker.runners):
            for family in ("current_predicate_index", "sequence_score", "sequence_complete"):
                for name, value in getattr(runner, family).items():
                    result[f"progress_{index}_{family}_{name}"] = value
        # This workflow has only the arrival predicate; reject unseen stateful
        # wrappers rather than silently omitting their counters.
        if any(runner._consecutive_step_requirements for runner in tracker.runners) or tracker._tracked_event_runners:
            raise ValueError("navigation continuation does not support additional temporal progress predicates")
    from .fluoroscopy_sources import bind_fluoroscopy_sources

    bind_fluoroscopy_sources(env)
    sensor = env.scene["fluoroscopy"]
    for name in ("_is_outdated", "_timestamp", "_timestamp_last_update", "_elapsed_since_update"):
        result[f"fluoroscopy{name}"] = wp.to_torch(getattr(sensor, name))
    result["fluoroscopy_frame_id"] = sensor._data.frame_id
    for name, value in sensor._data.output.items():
        result[f"fluoroscopy_output_{name}"] = value
    provider = sensor._carm_provider
    if hasattr(provider, "_pan_m"):
        if provider._pan_m is None:
            provider._pan_m = np.zeros(env.num_envs, np.float64)
        result["carm_pan"] = provider._pan_m
    return result


def ids_array(env, env_ids):
    if env_ids is None:
        return np.arange(env.num_envs)
    if isinstance(env_ids, slice):
        return np.arange(env.num_envs)[env_ids]
    return env_ids.detach().cpu().numpy() if isinstance(env_ids, torch.Tensor) else np.asarray(env_ids)


def capture(env, env_ids):
    env.i4h.sync_views()
    ids = ids_array(env, env_ids)
    host = {
        key: (value.detach().cpu().numpy() if isinstance(value, torch.Tensor) else value)[ids].copy()
        for key, value in host_arrays(env).items()
    }
    events = np.zeros((len(ids), 65536), np.uint8)
    if env.progress_tracker is not None:
        for row, world in enumerate(ids):
            encoded = json.dumps(
                [dict(asdict(event), env_idx=0) for event in env.progress_tracker._events[world]], sort_keys=True
            ).encode()
            if len(encoded) >= events.shape[1]:
                raise ValueError("navigation progress history exceeds continuation schema capacity")
            events[row, : len(encoded)] = np.frombuffer(encoded, np.uint8)
    host["progress_events"] = events
    host["fluoroscopy_dirty"] = np.full((len(ids), 1), env.scene["fluoroscopy"]._data_dirty, np.bool_)
    return dict(
        identity=np.repeat(configuration_identity(env)[None], len(ids), axis=0),
        numerical=env.i4h.capture_continuation(ids),
        host=host,
    )


def validate(env, bundle, env_ids=None):
    ids = ids_array(env, env_ids)
    if set(bundle) != {"identity", "numerical", "host"}:
        raise ValueError("navigation continuation schema mismatch")
    expected = np.repeat(configuration_identity(env)[None], len(ids), axis=0)
    actual = np.asarray(bundle["identity"])
    if actual.dtype != expected.dtype or actual.shape != expected.shape or actual.tobytes() != expected.tobytes():
        raise ValueError("navigation continuation backend/schema/source/configuration/anatomy mismatch")
    env.i4h.validate_continuation(bundle["numerical"], ids)
    destinations = host_arrays(env)
    if set(destinations) | {"progress_events", "fluoroscopy_dirty"} != set(bundle["host"]):
        raise ValueError(
            f"navigation host continuation layout mismatch: "
            f"missing={set(destinations)-set(bundle['host'])}, "
            f"extra={set(bundle['host'])-set(destinations)-{'progress_events', 'fluoroscopy_dirty'}}"
        )
    for key, destination in destinations.items():
        like = destination.detach().cpu().numpy() if isinstance(destination, torch.Tensor) else destination
        source = np.asarray(bundle["host"][key])
        if source.dtype != like.dtype or source.shape != (len(ids), *like.shape[1:]):
            raise ValueError(f"navigation host shape/dtype mismatch: {key}")
    dirty = np.asarray(bundle["host"]["fluoroscopy_dirty"])
    if dirty.dtype != np.bool_ or dirty.shape != (len(ids), 1) or (len(ids) and not np.all(dirty == dirty[0])):
        raise ValueError("navigation fluoroscopy dirty flag layout mismatch")
    events = np.asarray(bundle["host"]["progress_events"])
    if events.dtype != np.uint8 or events.shape != (len(ids), 65536):
        raise ValueError("navigation progress event shape/dtype mismatch")
    from isaaclab_arena.progress_tracking.progress_tracker import PredicateEvent

    for row in events:
        data = row.tobytes().split(b"\0", 1)[0]
        for event in json.loads(data or b"[]"):
            PredicateEvent(**event)
    return ids, destinations


def restore(env, bundle, env_ids=None):
    ids, destinations = validate(env, bundle, env_ids)
    # Host scene assets are restored by reset_to before this boundary. Flush the
    # host's deferred solver cache reset before writing numerical history.
    if env.i4h.physics.coupled is not None:
        robot = env.scene["robot"]
        device_ids = torch.as_tensor(ids, device=env.device)
        robot.write_joint_state_to_sim(
            torch.as_tensor(bundle["host"]["robot_joint_pos"], device=env.device),
            torch.as_tensor(bundle["host"]["robot_joint_vel"], device=env.device),
            env_ids=device_ids,
        )
    env.sim.forward()
    device_ids = torch.as_tensor(ids, device=env.device)
    for key, destination in destinations.items():
        if isinstance(destination, torch.Tensor):
            destination[device_ids] = torch.as_tensor(bundle["host"][key], device=destination.device)
        else:
            destination[ids] = bundle["host"][key]
    env.action_manager.get_term("carm_orbit")._write_pose()
    env.i4h.restore_continuation(bundle["numerical"], ids)
    env.i4h.order_inputs()
    env.i4h.sync_views()
    if env.progress_tracker is not None:
        from isaaclab_arena.progress_tracking.progress_tracker import PredicateEvent

        for row, world in enumerate(ids):
            encoded = bundle["host"]["progress_events"][row].tobytes().split(b"\0", 1)[0]
            env.progress_tracker._events[world] = [
                PredicateEvent(**dict(event, env_idx=int(world))) for event in json.loads(encoded or b"[]")
            ]
    sensor = env.scene["fluoroscopy"]
    dirty = bool(bundle["host"]["fluoroscopy_dirty"].any())
    sensor._data_dirty = dirty if len(ids) == env.num_envs else sensor._data_dirty or dirty
    env._task_progress_step = env.common_step_counter
    catheter = env.scene["catheter"]
    catheter._tip_bend_rad = env.action_manager.get_term("catheter")._tip_bend_angle
    catheter._refresh_positions()
    from .navigation_reward import TIP_STATE_CACHE_ATTR

    if hasattr(env, TIP_STATE_CACHE_ATTR):
        delattr(env, TIP_STATE_CACHE_ATTR)

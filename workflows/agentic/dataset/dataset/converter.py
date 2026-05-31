# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import inspect
import json
import logging
import os
import shutil
from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
from common.config import get_environment_config, get_robot_config
from tqdm import tqdm

logger = logging.getLogger("dataset")


def _resize_frame(img, target_hw):
    """Resize an HxWx3 uint8 frame to (H, W); no-op if already that size."""
    import cv2  # noqa: PLC0415

    if tuple(img.shape[:2]) == (int(target_hw[0]), int(target_hw[1])):
        return img
    return cv2.resize(img, (int(target_hw[1]), int(target_hw[0])), interpolation=cv2.INTER_AREA)


_WORKFLOW_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class RecordingSchema:
    camera_keys: list[str]
    action_dim: int
    state_dim: int
    camera_shape: tuple[int, int, int]


def nonnegative_int(value: str) -> int:
    parsed = int(value)
    if parsed < 0:
        raise argparse.ArgumentTypeError("must be zero or a positive integer")
    return parsed


def parse_split(value: str) -> list[tuple[str, int, int]]:
    if not value:
        return []
    out = []
    for piece in (part.strip() for part in value.split(",") if part.strip()):
        try:
            name, span = piece.split(":")
            start, end = span.split("-")
            out.append((name.strip(), int(start), int(end)))
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"bad split entry {piece!r}: {exc}") from exc
    return out


def apply_env_defaults(args: argparse.Namespace, parser: argparse.ArgumentParser) -> None:
    args.dataset_config = {}
    if args.env:
        env_config = _environment_block(args.env)
        robot_config = env_config.get("robot", {})
        policy_config = env_config.get("policy", {})
        args.dataset_config = env_config.get("dataset", {})
        args.repo_id = args.repo_id or args.dataset_config.get("repo_id") or policy_config.get("repo_id")
        args.repo_id = args.repo_id or f"local/{args.env}"
        args.task_description = (
            args.task_description
            or args.dataset_config.get("task_description")
            or policy_config.get("task_description")
            or policy_config.get("language_instruction")
        )
        args.joint_space = args.joint_space or args.dataset_config.get("joint_space")
        args.action_source_frame = args.action_source_frame or args.dataset_config.get("action_source_frame")
        args.action_output_frame = args.action_output_frame or args.dataset_config.get("action_output_frame")
        args.home_joint_pos_rad = (
            args.home_joint_pos_rad
            or args.dataset_config.get("home_joint_pos_rad")
            or robot_config.get("home_joint_pos_rad")
            or env_config.get("home_joint_pos_rad")
        )
        args.robot_type = args.robot_type or args.dataset_config.get("robot_type") or robot_config.get("type")
        # Honor per-env state_obs_key override (e.g. ultrasound uses joint_pos_rel).
        env_state_obs_key = args.dataset_config.get("state_obs_key")
        if env_state_obs_key and args.state_obs_key == "joint_pos":
            args.state_obs_key = env_state_obs_key
        if args.image_size is None:
            size = args.dataset_config.get("image_size") or policy_config.get("image_size")
            if size and len(size) == 2:
                args.image_size = (int(size[0]), int(size[1]))
    elif not args.repo_id:
        parser.error("--repo-id is required unless --env is provided")

    args.task_description = args.task_description or "Perform the recorded manipulation task."
    args.joint_space = args.joint_space or "raw"
    args.action_source_frame = args.action_source_frame or "absolute"
    args.action_output_frame = args.action_output_frame or "absolute"
    args.home_joint_pos_rad = _home_pose_from_config(args.home_joint_pos_rad)
    args.robot_type = args.robot_type or "so101"
    args.cameras = _csv(args.cameras)
    args.action_names = _csv(args.action_names) or _names_from_config(args.dataset_config.get("action_names"))
    args.state_names = _csv(args.state_names) or _names_from_config(args.dataset_config.get("state_names"))
    args.state_split = args.state_split or _split_from_config(args.dataset_config.get("state_split"))
    args.action_split = args.action_split or _split_from_config(args.dataset_config.get("action_split"))
    if args.skip_frames is None:
        args.skip_frames = int(args.dataset_config.get("skip_frames", 0))
    if args.image_size is not None:
        args.image_size = (int(args.image_size[0]), int(args.image_size[1]))


def convert_hdf5_to_lerobot(args: argparse.Namespace) -> None:
    if not os.path.exists(args.hdf5_path):
        raise FileNotFoundError(args.hdf5_path)

    from lerobot.common.constants import HF_LEROBOT_HOME
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset

    _use_video_codec(args.video_codec)
    dataset_root = HF_LEROBOT_HOME / args.repo_id
    if args.overwrite and dataset_root.exists():
        logger.info("deleting existing dataset: %s", dataset_root)
        shutil.rmtree(dataset_root)

    converter = _dataset_converter(args.dataset_config or {})
    if converter == "locomanip_g1":
        _convert_locomanip_g1(args, LeRobotDataset)
        if args.push_to_hub:
            logger.warning("push_to_hub not supported for locomanip_g1 converter")
        logger.info("done.")
        return
    if converter == "trocar_g1":
        _convert_trocar_g1(args, LeRobotDataset)
        if args.push_to_hub:
            logger.warning("push_to_hub not supported for trocar_g1 converter")
        logger.info("done.")
        return

    with h5py.File(args.hdf5_path, "r") as file:
        demos, schema = _read_schema(file, args)
        dataset = _create_dataset(LeRobotDataset, args, schema)
        saved = _write_episodes(dataset, demos, args, schema.camera_keys)

    logger.info("saved %d episodes to %s", saved, dataset.root)
    _warn_degenerate_action_stats(dataset)
    _write_modality_json(dataset.root, args, schema)
    if args.push_to_hub:
        dataset.push_to_hub()
    logger.info("done.")


# locomanip_g1 converter for G1 loco-manipulation envs.


def _convert_locomanip_g1(args: argparse.Namespace, LeRobotDataset) -> None:
    cfg = args.dataset_config or {}
    state_obs_key = args.state_obs_key or "robot_joint_pos"
    action_key = cfg.get("action_key", "processed_actions")
    camera_mappings = cfg.get("camera_mappings") or {}
    if not camera_mappings:
        raise ValueError(
            "locomanip_g1 converter requires dataset.camera_mappings, e.g. "
            "{robot_head_cam: observation.images.ego_view}"
        )
    modality_template = _resolve_template(cfg.get("modality_template_path"))
    if modality_template is None or not modality_template.exists():
        raise ValueError(
            "locomanip_g1 converter requires dataset.modality_template_path " f"(resolved: {modality_template})"
        )
    teleop_zero = cfg.get("teleop_zero_fill") or [
        {"name": "teleop.base_height_command", "dim": 1, "dtype": "float32"},
        {"name": "teleop.navigate_command", "dim": 3, "dtype": "float32"},
        {"name": "teleop.torso_orientation_rpy_command", "dim": 3, "dtype": "float32"},
    ]
    teleop_sources = dict(cfg.get("teleop_sources") or {})
    state_dim = int(cfg.get("state_dim", 43))
    action_dim = int(cfg.get("action_dim", 43))

    with h5py.File(args.hdf5_path, "r") as file:
        demos = _open_demos(file)
        if not demos:
            raise ValueError(f"no successful demos in {args.hdf5_path}")

        if not teleop_sources:
            first_demo_group = demos[0][1]
            raw_actions = first_demo_group.get("actions")
            if raw_actions is not None and raw_actions.ndim == 2 and raw_actions.shape[1] == 23:
                teleop_sources = {
                    "teleop.navigate_command": {"from": "actions", "start": 16, "end": 19},
                    "teleop.base_height_command": {"from": "actions", "start": 19, "end": 20},
                    "teleop.torso_orientation_rpy_command": {"from": "actions", "start": 20, "end": 23},
                }
                logger.info(
                    "auto-defaulted teleop_sources for 23D keyboard_23d HDF5; "
                    "set dataset.teleop_sources in the env YAML to override."
                )
        _, first_group = demos[0]

        first_obs = first_group["obs"]
        sim_cams = list(camera_mappings.keys())
        for cam in sim_cams:
            if cam not in first_obs:
                raise KeyError(f"camera_mappings references obs/{cam} but it is missing in HDF5")

        features = {
            "action": {
                "dtype": "float32",
                "shape": (action_dim,),
                "names": _g1_43dof_names(),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (state_dim,),
                "names": _g1_43dof_names(),
            },
        }
        for entry in teleop_zero:
            features[entry["name"]] = {
                "dtype": entry.get("dtype", "float32"),
                "shape": (int(entry["dim"]),),
                "names": [f"{entry['name']}.{i}" for i in range(int(entry["dim"]))],
            }
        for cam, lerobot_key in camera_mappings.items():  # noqa: B007
            cam_suffix = lerobot_key.split("observation.images.", 1)[-1]
            # Resize all cameras to the configured image_size so the dataset matches
            # the modality/policy config; fall back to each camera's native shape.
            cam_shape = (
                (int(args.image_size[0]), int(args.image_size[1]), 3)
                if args.image_size
                else tuple(first_obs[cam].shape[1:])
            )
            features[f"observation.images.{cam_suffix}"] = _video_feature(
                cam_shape,
                args.video_codec,
                args.fps,
            )

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            robot_type=args.robot_type or "g1",
            features=features,
            **_lerobot_create_kwargs(LeRobotDataset.create, args),
        )

        saved = 0
        for name, group in tqdm(demos, desc="demos"):
            try:
                state = np.asarray(group[f"obs/{state_obs_key}"], dtype=np.float32)
                action = np.asarray(group[action_key], dtype=np.float32)
                video_arrays = {cam: np.asarray(group[f"obs/{cam}"]) for cam in sim_cams}
            except KeyError as exc:
                logger.info("skip %s (missing %s)", name, exc)
                continue

            teleop_source_arrays: dict[str, np.ndarray] = {}
            for source_name in {spec["from"] for spec in teleop_sources.values() if spec.get("from")}:
                if source_name in group:
                    teleop_source_arrays[source_name] = np.asarray(group[source_name], dtype=np.float32)
                elif f"obs/{source_name}" in group:
                    teleop_source_arrays[source_name] = np.asarray(group[f"obs/{source_name}"], dtype=np.float32)
                else:
                    logger.warning(
                        "%s: teleop_sources references '%s' but it is missing from HDF5; "
                        "the columns mapped to that source will be zero-filled",
                        name,
                        source_name,
                    )

            n = min(state.shape[0], action.shape[0], *(v.shape[0] for v in video_arrays.values()))
            if state.shape[-1] != state_dim:
                raise ValueError(f"{name}: state has {state.shape[-1]} dims, expected {state_dim}")
            if action.shape[-1] != action_dim:
                raise ValueError(f"{name}: action has {action.shape[-1]} dims, expected {action_dim}")

            for index in range(args.skip_frames, n):
                frame = {
                    "action": action[index],
                    "observation.state": state[index],
                }
                for entry in teleop_zero:
                    dim = int(entry["dim"])
                    dtype = np.dtype(entry.get("dtype", "float32"))
                    spec = teleop_sources.get(entry["name"])
                    source_arr = teleop_source_arrays.get(spec["from"]) if spec and spec.get("from") else None
                    if spec and source_arr is not None:
                        start = int(spec.get("start", 0))
                        end = int(spec.get("end", start + dim))
                        value = source_arr[index, start:end].astype(dtype, copy=False)
                        if value.shape[0] != dim:
                            raise ValueError(
                                f"{name}: teleop_sources['{entry['name']}'] yielded {value.shape[0]} "
                                f"values, expected {dim}"
                            )
                        frame[entry["name"]] = value
                    else:
                        frame[entry["name"]] = np.zeros(dim, dtype=dtype)
                for cam, lerobot_key in camera_mappings.items():  # noqa: B007
                    cam_suffix = lerobot_key.split("observation.images.", 1)[-1]
                    img = video_arrays[cam][index]
                    if args.image_size:
                        img = _resize_frame(img, args.image_size)
                    frame[f"observation.images.{cam_suffix}"] = img
                dataset.add_frame(frame=frame, task=args.task_description)
            dataset.save_episode()
            saved += 1

    logger.info("saved %d episodes to %s", saved, dataset.root)

    _warn_degenerate_action_stats(dataset)

    meta_dir = Path(dataset.root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(modality_template, meta_dir / "modality.json")
    logger.info("copied modality.json from %s", modality_template)


def _warn_degenerate_action_stats(dataset) -> None:
    """Warn when any action/teleop column has zero variance."""
    stats = getattr(dataset, "stats", None)
    if not stats:
        return
    degenerate: list[tuple[str, int]] = []
    for key, stat in stats.items():
        if not (key == "action" or key.startswith("action.") or key.startswith("teleop.")):
            continue
        std = stat.get("std") if isinstance(stat, dict) else None
        if std is None:
            continue
        std_arr = np.asarray(std, dtype=np.float32).reshape(-1)
        if std_arr.size == 0:
            continue
        zero_slots = [int(i) for i, v in enumerate(std_arr) if float(v) == 0.0]
        if zero_slots:
            degenerate.append((key, len(zero_slots), std_arr.size))
    if not degenerate:
        return
    logger.warning(
        "%d action/teleop column(s) have zero variance across all written "
        "episodes. GR00T's policy head will collapse to the unnormalized "
        "midpoint for these channels and the robot will appear to ignore "
        "them at inference. Either add varied demos for the affected "
        "channels, inject Gaussian noise in the converter, or remove the "
        "key from the modality config so the head isn't trained.",
        len(degenerate),
    )
    for key, n_zero, n_total in degenerate:
        logger.warning("  %s: %d/%d slots have std=0", key, n_zero, n_total)


def _resolve_template(path: str | None) -> Path | None:
    if not path:
        return None
    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = (_WORKFLOW_ROOT / candidate).resolve()
    return candidate


def _g1_43dof_names() -> list[str]:
    """Joint names in the IsaacLab-Arena gr00t_43dof_joint_space.yaml order."""
    return [
        "left_hip_pitch_joint",
        "left_hip_roll_joint",
        "left_hip_yaw_joint",
        "left_knee_joint",
        "left_ankle_pitch_joint",
        "left_ankle_roll_joint",
        "right_hip_pitch_joint",
        "right_hip_roll_joint",
        "right_hip_yaw_joint",
        "right_knee_joint",
        "right_ankle_pitch_joint",
        "right_ankle_roll_joint",
        "waist_yaw_joint",
        "waist_roll_joint",
        "waist_pitch_joint",
        "left_shoulder_pitch_joint",
        "left_shoulder_roll_joint",
        "left_shoulder_yaw_joint",
        "left_elbow_joint",
        "left_wrist_roll_joint",
        "left_wrist_pitch_joint",
        "left_wrist_yaw_joint",
        "left_hand_index_0_joint",
        "left_hand_index_1_joint",
        "left_hand_middle_0_joint",
        "left_hand_middle_1_joint",
        "left_hand_thumb_0_joint",
        "left_hand_thumb_1_joint",
        "left_hand_thumb_2_joint",
        "right_shoulder_pitch_joint",
        "right_shoulder_roll_joint",
        "right_shoulder_yaw_joint",
        "right_elbow_joint",
        "right_wrist_roll_joint",
        "right_wrist_pitch_joint",
        "right_wrist_yaw_joint",
        "right_hand_index_0_joint",
        "right_hand_index_1_joint",
        "right_hand_middle_0_joint",
        "right_hand_middle_1_joint",
        "right_hand_thumb_0_joint",
        "right_hand_thumb_1_joint",
        "right_hand_thumb_2_joint",
    ]


def _environment_block(env_id: str) -> dict:
    return get_environment_config(env_id)


def _csv(value: str | None) -> list[str] | None:
    if value is None:
        return None
    return [item.strip() for item in value.split(",") if item.strip()]


def _names_from_config(value) -> list[str] | None:
    return None if value is None else [str(name) for name in value]


def _split_from_config(value) -> list[tuple[str, int, int]] | None:
    return None if value is None else [(str(name), int(start), int(end)) for name, start, end in value]


def _dataset_converter(config: dict) -> str:
    if config.get("state_body_key") or config.get("state_dex3_key"):
        return "trocar_g1"
    if (
        config.get("state_obs_key") == "robot_joint_pos"
        and config.get("action_key") == "processed_actions"
        and config.get("state_dim") == 43
        and config.get("action_dim") == 43
    ):
        return "locomanip_g1"
    return "default"


def _home_pose_from_config(value) -> np.ndarray | None:
    if value is None or value == "":
        return None
    if isinstance(value, str):
        values = [float(item.strip()) for item in value.split(",") if item.strip()]
    else:
        values = [float(item) for item in value]
    return np.asarray(values, dtype=np.float32)


def _read_schema(file: h5py.File, args: argparse.Namespace) -> tuple[list[tuple[str, h5py.Group]], RecordingSchema]:
    demos = _open_demos(file)
    if not demos:
        raise ValueError(f"no demos in {args.hdf5_path}")

    _, first_group = demos[0]
    camera_keys = _infer_camera_keys(first_group, args.cameras)
    if not camera_keys:
        raise ValueError("no camera arrays found in HDF5; pass --cameras explicitly or record with cameras enabled")

    action_dim, state_dim = _infer_dims(first_group, args)
    camera_shape = tuple(first_group[f"obs/{camera_keys[0]}"].shape[1:])
    logger.info(
        "schema: action_dim=%s state_dim=%s cameras=%s camera_shape=%s",
        action_dim,
        state_dim,
        camera_keys,
        camera_shape,
    )
    return demos, RecordingSchema(camera_keys, action_dim, state_dim, camera_shape)


def _open_demos(file: h5py.File) -> list[tuple[str, h5py.Group]]:
    demos = []
    for name, group in file["data"].items():
        if hasattr(group, "keys") and not ("success" in group.attrs and not group.attrs["success"]):
            demos.append((name, group))
        else:
            logger.info("skip %s (not successful)", name)
    return demos


def _infer_camera_keys(group: h5py.Group, override: list[str] | None) -> list[str]:
    if override:
        return list(override)
    obs = group.get("obs")
    if obs is None:
        return []
    return [
        name
        for name, ds in obs.items()
        if hasattr(ds, "shape") and ds.ndim == 4 and ds.shape[-1] == 3 and ds.dtype == np.uint8
    ]


def _infer_dims(group: h5py.Group, args: argparse.Namespace) -> tuple[int, int]:
    actions = np.asarray(group["obs/actions"]) if "obs/actions" in group else np.asarray(group["actions"])
    state_path = f"obs/{args.state_obs_key}"
    state = np.asarray(group[state_path]) if state_path in group else actions
    action_dim = args.action_dim if args.action_dim is not None else actions.shape[-1]
    state_dim = args.state_dim if args.state_dim is not None else state.shape[-1]
    return action_dim, state_dim


def _create_dataset(LeRobotDataset, args: argparse.Namespace, schema: RecordingSchema):
    return LeRobotDataset.create(
        repo_id=args.repo_id,
        fps=args.fps,
        robot_type=args.robot_type,
        features=_features(args, schema),
        **_lerobot_create_kwargs(LeRobotDataset.create, args),
    )


def _features(args: argparse.Namespace, schema: RecordingSchema) -> dict:
    features = {
        "action": {
            "dtype": "float32",
            "shape": (schema.action_dim,),
            "names": args.action_names or [f"action.{i}" for i in range(schema.action_dim)],
        },
        "observation.state": {
            "dtype": "float32",
            "shape": (schema.state_dim,),
            "names": args.state_names or [f"state.{i}" for i in range(schema.state_dim)],
        },
    }
    for cam in schema.camera_keys:
        features[f"observation.images.{cam}"] = _video_feature(schema.camera_shape, args.video_codec, args.fps)
    return features


def _video_feature(shape: tuple[int, int, int], video_codec: str, fps: int) -> dict:
    return {
        "dtype": "video",
        "shape": list(shape),
        "names": ["height", "width", "channels"],
        "video_info": {
            "video.height": shape[0],
            "video.width": shape[1],
            "video.codec": video_codec,
            "video.pix_fmt": "yuv420p",
            "video.is_depth_map": False,
            "video.fps": float(fps),
            "video.channels": 3,
            "has_audio": False,
        },
    }


def _lerobot_create_kwargs(create, args: argparse.Namespace) -> dict:
    supported = inspect.signature(create).parameters
    requested = {
        "image_writer_processes": args.image_writer_processes,
        "image_writer_threads": args.image_writer_threads,
    }
    return {name: value for name, value in requested.items() if name in supported}


def _write_episodes(
    dataset,
    demos: list[tuple[str, h5py.Group]],
    args: argparse.Namespace,
    camera_keys: list[str],
) -> int:
    saved = 0
    state_path = f"obs/{args.state_obs_key}"
    for name, group in tqdm(demos, desc="demos"):
        try:
            actions = np.asarray(group["obs/actions"])
            joint_pos = np.asarray(group[state_path])
            camera_data = {cam: np.asarray(group[f"obs/{cam}"]) for cam in camera_keys}
        except KeyError as exc:
            logger.info("skip %s (missing %s)", name, exc)
            continue

        actions, joint_pos = _apply_joint_space(actions, joint_pos, args)
        n = actions.shape[0]
        assert n == joint_pos.shape[0]
        for cam, arr in camera_data.items():
            assert arr.shape[0] == n, f"{cam}: {arr.shape[0]} vs actions {n}"

        for index in tqdm(range(args.skip_frames, n), desc=f"{name} frames", leave=False):
            frame = {"action": actions[index], "observation.state": joint_pos[index]}
            for cam in camera_keys:
                frame[f"observation.images.{cam}"] = camera_data[cam][index]
            dataset.add_frame(frame=frame, task=args.task_description)
        dataset.save_episode()
        saved += 1
    return saved


def _apply_joint_space(
    actions: np.ndarray,
    joint_pos: np.ndarray,
    args: argparse.Namespace,
) -> tuple[np.ndarray, np.ndarray]:
    actions = _actions_to_absolute_source_frame(actions, args)
    if args.joint_space == "raw":
        return (
            _actions_from_absolute_frame(actions, args, target_space=False).astype(np.float32),
            joint_pos.astype(np.float32),
        )
    if args.joint_space != "remap":
        raise ValueError(f"unknown --joint-space: {args.joint_space}")

    source_range, target_range = _joint_ranges(args)
    if not source_range or not target_range:
        raise ValueError(
            "joint_space=remap requires dataset.joint_source_range/joint_target_range "
            f"or known robot_type ranges; got robot_type={args.robot_type!r}"
        )

    action_source = _convert_source_unit(actions, args.dataset_config.get("joint_source_unit", "raw"))
    state_source = _convert_source_unit(joint_pos, args.dataset_config.get("joint_source_unit", "raw"))
    action_target = _remap_joint_range(action_source, source_range, target_range)
    state_target = _remap_joint_range(state_source, source_range, target_range)
    action_target = _actions_from_absolute_frame(action_target, args, target_space=True)
    return action_target.astype(np.float32), state_target.astype(np.float32)


def _actions_to_absolute_source_frame(actions: np.ndarray, args: argparse.Namespace) -> np.ndarray:
    source_frame = args.action_source_frame
    if source_frame not in ("absolute", "relative_to_home"):
        raise ValueError(f"unknown action_source_frame: {source_frame}")
    actions = np.asarray(actions, dtype=np.float32)
    if source_frame == "absolute":
        return actions
    return actions + _action_home_pose(args, target_space=False)


def _actions_from_absolute_frame(actions: np.ndarray, args: argparse.Namespace, *, target_space: bool) -> np.ndarray:
    output_frame = args.action_output_frame
    if output_frame not in ("absolute", "relative_to_home"):
        raise ValueError(f"unknown action_output_frame: {output_frame}")
    actions = np.asarray(actions, dtype=np.float32)
    if output_frame == "absolute":
        return actions
    return actions - _action_home_pose(args, target_space=target_space)


def _action_home_pose(args: argparse.Namespace, *, target_space: bool) -> np.ndarray:
    home = args.home_joint_pos_rad
    if home is None:
        raise ValueError(
            f"action frame conversion {args.action_source_frame}->{args.action_output_frame} "
            "requires home_joint_pos_rad"
        )
    home = np.asarray(home, dtype=np.float32)
    if target_space:
        source_range, target_range = _joint_ranges(args)
        if not source_range or not target_range:
            raise ValueError(
                "target-space home conversion requires joint_source_range/joint_target_range "
                f"or known robot_type ranges; got robot_type={args.robot_type!r}"
            )
        source_unit = args.dataset_config.get("joint_source_unit", "raw")
        home = _convert_source_unit(home, source_unit)
        home = _remap_joint_range(home, source_range, target_range)
    return home.reshape(1, -1)


def _joint_ranges(args: argparse.Namespace) -> tuple[list | tuple | None, list | tuple | None]:
    source_range = args.dataset_config.get("joint_source_range")
    target_range = args.dataset_config.get("joint_target_range")
    if source_range and target_range:
        return source_range, target_range
    if args.robot_type:
        robot_config = get_robot_config(args.robot_type)
        return robot_config.isaaclab_joint_pos_limit_range, robot_config.lerobot_joint_pos_limit_range
    return source_range, target_range


def _convert_source_unit(values: np.ndarray, source_unit: str) -> np.ndarray:
    values = np.asarray(values, dtype=np.float32)
    if source_unit == "rad":
        return values / np.pi * 180.0
    if source_unit in ("raw", "deg"):
        return values
    raise ValueError(f"unknown dataset.joint_source_unit: {source_unit}")


def _remap_joint_range(joint_pos: np.ndarray, source_range: list, target_range: list) -> np.ndarray:
    source = np.asarray(source_range, dtype=np.float32)
    target = np.asarray(target_range, dtype=np.float32)
    if source.shape != target.shape:
        raise ValueError(f"source and target joint ranges must match, got {source.shape} and {target.shape}")
    if joint_pos.shape[-1] != source.shape[0]:
        raise ValueError(f"expected {source.shape[0]} joint positions, got {joint_pos.shape[-1]}")
    return (joint_pos - source[:, 0]) / (source[:, 1] - source[:, 0]) * (target[:, 1] - target[:, 0]) + target[:, 0]


def _write_modality_json(dataset_root: Path, args: argparse.Namespace, schema: RecordingSchema) -> None:
    state_split = args.state_split or [("state", 0, schema.state_dim)]
    action_split = args.action_split or [("action", 0, schema.action_dim)]
    meta_dir = dataset_root / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    modality = {
        "state": {name: {"start": start, "end": end} for name, start, end in state_split},
        "action": {name: {"start": start, "end": end} for name, start, end in action_split},
        "video": {cam: {"original_key": f"observation.images.{cam}"} for cam in schema.camera_keys},
        "annotation": {"human.task_description": {"original_key": "task_index"}},
    }
    with open(meta_dir / "modality.json", "w") as file:
        json.dump(modality, file, indent=4)


def _use_video_codec(video_codec: str) -> None:
    from lerobot.common.datasets import lerobot_dataset
    from lerobot.common.datasets.video_utils import encode_video_frames

    def encode_with_codec(imgs_dir, video_path, fps, overwrite=False):
        return encode_video_frames(imgs_dir, video_path, fps, vcodec=video_codec, overwrite=overwrite)

    lerobot_dataset.encode_video_frames = encode_with_codec


# trocar_g1 converter for assemble-trocar.

_TROCAR_STATE_28_NAMES = [
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
    "left_hand_thumb_0_joint",
    "left_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "left_hand_middle_0_joint",
    "left_hand_middle_1_joint",
    "left_hand_index_0_joint",
    "left_hand_index_1_joint",
    "right_hand_thumb_0_joint",
    "right_hand_thumb_1_joint",
    "right_hand_thumb_2_joint",
    "right_hand_middle_0_joint",
    "right_hand_middle_1_joint",
    "right_hand_index_0_joint",
    "right_hand_index_1_joint",
]

_TROCAR_RECORDED_ACTION_43_NAMES = (
    "left_hip_pitch_joint",
    "right_hip_pitch_joint",
    "waist_yaw_joint",
    "left_hip_roll_joint",
    "right_hip_roll_joint",
    "waist_roll_joint",
    "left_hip_yaw_joint",
    "right_hip_yaw_joint",
    "waist_pitch_joint",
    "left_knee_joint",
    "right_knee_joint",
    "left_shoulder_pitch_joint",
    "right_shoulder_pitch_joint",
    "left_ankle_pitch_joint",
    "right_ankle_pitch_joint",
    "left_shoulder_roll_joint",
    "right_shoulder_roll_joint",
    "left_ankle_roll_joint",
    "right_ankle_roll_joint",
    "left_shoulder_yaw_joint",
    "right_shoulder_yaw_joint",
    "left_elbow_joint",
    "right_elbow_joint",
    "left_wrist_roll_joint",
    "right_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "right_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    "right_wrist_yaw_joint",
    "left_hand_index_0_joint",
    "left_hand_middle_0_joint",
    "left_hand_thumb_0_joint",
    "right_hand_index_0_joint",
    "right_hand_middle_0_joint",
    "right_hand_thumb_0_joint",
    "left_hand_index_1_joint",
    "left_hand_middle_1_joint",
    "left_hand_thumb_1_joint",
    "right_hand_index_1_joint",
    "right_hand_middle_1_joint",
    "right_hand_thumb_1_joint",
    "left_hand_thumb_2_joint",
    "right_hand_thumb_2_joint",
)

_TROCAR_BODY_COL_INDICES = (
    0,
    3,
    6,
    9,
    13,
    17,
    1,
    4,
    7,
    10,
    14,
    18,
    2,
    5,
    8,
    11,
    15,
    19,
    21,
    23,
    25,
    27,
    12,
    16,
    20,
    22,
    24,
    26,
    28,
)
_TROCAR_DEX3_COL_INDICES = (31, 37, 41, 30, 36, 29, 35, 34, 40, 42, 33, 39, 32, 38)

_TROCAR_RAW_ACTION_DELTA = np.zeros(28, dtype=np.float64)
_TROCAR_RAW_ACTION_DELTA[3] = 0.3  # left_elbow
_TROCAR_RAW_ACTION_DELTA[10] = 0.3  # right_elbow


def _build_trocar_index_tables() -> tuple[list[int], list[int], list[int], list[int]]:
    """Return action and state remap indices for assemble-trocar."""
    name_to_idx_43 = {name: i for i, name in enumerate(_TROCAR_RECORDED_ACTION_43_NAMES)}
    action_43_to_28 = [name_to_idx_43[name] for name in _TROCAR_STATE_28_NAMES]

    body_joint_to_col = {jid: i for i, jid in enumerate(_TROCAR_BODY_COL_INDICES)}
    body_left_arm_cols = [body_joint_to_col[j] for j in range(15, 22)]
    body_right_arm_cols = [body_joint_to_col[j] for j in range(22, 29)]

    dex3_joint_to_col = {jid: i for i, jid in enumerate(_TROCAR_DEX3_COL_INDICES)}
    dex3_left_hand_cols = [dex3_joint_to_col[j] for j in range(29, 36)]
    dex3_right_hand_cols = [dex3_joint_to_col[j] for j in range(36, 43)]

    return action_43_to_28, body_left_arm_cols, body_right_arm_cols, dex3_left_hand_cols + dex3_right_hand_cols


def _convert_trocar_g1(args: argparse.Namespace, LeRobotDataset) -> None:
    cfg = args.dataset_config or {}
    camera_mappings = cfg.get("camera_mappings") or {}
    if not camera_mappings:
        raise ValueError("trocar_g1 converter requires dataset.camera_mappings")
    modality_template = _resolve_template(cfg.get("modality_template_path"))
    if modality_template is None or not modality_template.exists():
        raise ValueError(
            "trocar_g1 converter requires dataset.modality_template_path " f"(resolved: {modality_template})"
        )
    state_body_key = cfg.get("state_body_key", "robot_joint_state")
    state_dex3_key = cfg.get("state_dex3_key", "robot_dex3_joint_state")
    action_key = cfg.get("action_key", "processed_actions")

    action_43_to_28, body_la, body_ra, dex3_cols = _build_trocar_index_tables()

    with h5py.File(args.hdf5_path, "r") as file:
        demos = _open_demos(file)
        if not demos:
            raise ValueError(f"no successful demos in {args.hdf5_path}")
        _, first_group = demos[0]
        first_obs = first_group["obs"]
        sim_cams = list(camera_mappings.keys())
        for cam in sim_cams:
            if cam not in first_obs:
                raise KeyError(f"camera_mappings references obs/{cam} but it is missing in HDF5")

        features = {
            "action": {
                "dtype": "float32",
                "shape": (28,),
                "names": list(_TROCAR_STATE_28_NAMES),
            },
            "observation.state": {
                "dtype": "float32",
                "shape": (28,),
                "names": list(_TROCAR_STATE_28_NAMES),
            },
        }
        for cam, lerobot_key in camera_mappings.items():  # noqa: B007
            cam_suffix = lerobot_key.split("observation.images.", 1)[-1]
            # Resize all cameras to the configured image_size so the dataset matches
            # the modality/policy config; fall back to each camera's native shape.
            cam_shape = (
                (int(args.image_size[0]), int(args.image_size[1]), 3)
                if args.image_size
                else tuple(first_obs[cam].shape[1:])
            )
            features[f"observation.images.{cam_suffix}"] = _video_feature(
                cam_shape,
                args.video_codec,
                args.fps,
            )

        dataset = LeRobotDataset.create(
            repo_id=args.repo_id,
            fps=args.fps,
            robot_type=args.robot_type or "g1",
            features=features,
            **_lerobot_create_kwargs(LeRobotDataset.create, args),
        )

        saved = 0
        for name, group in tqdm(demos, desc="demos"):
            try:
                state_body = np.asarray(group[f"obs/{state_body_key}"])
                state_dex3 = np.asarray(group[f"obs/{state_dex3_key}"])
                action_full = np.asarray(group[action_key])
                video_arrays = {cam: np.asarray(group[f"obs/{cam}"]) for cam in sim_cams}
            except KeyError as exc:
                logger.info("skip %s (missing %s)", name, exc)
                continue

            if state_body.shape[-1] != 87:
                raise ValueError(f"{name}: robot_joint_state has {state_body.shape[-1]} dims, expected 87")
            if state_dex3.shape[-1] != 14:
                raise ValueError(f"{name}: robot_dex3_joint_state has {state_dex3.shape[-1]} dims, expected 14")
            if action_full.shape[-1] != 43:
                raise ValueError(f"{name}: processed_actions has {action_full.shape[-1]} dims, expected 43")

            state = np.concatenate(
                [
                    state_body[:-1, body_la],
                    state_body[:-1, body_ra],
                    state_dex3[:-1, dex3_cols[:7]],
                    state_dex3[:-1, dex3_cols[7:]],
                ],
                axis=1,
            ).astype(np.float32)
            action = action_full[:-1, action_43_to_28].astype(np.float32) + _TROCAR_RAW_ACTION_DELTA.astype(np.float32)
            n = state.shape[0]

            for index in range(args.skip_frames, n):
                frame = {
                    "action": action[index],
                    "observation.state": state[index],
                }
                for cam, lerobot_key in camera_mappings.items():  # noqa: B007
                    cam_suffix = lerobot_key.split("observation.images.", 1)[-1]
                    # Align video to state length (drop the last frame to match T-1).
                    frame[f"observation.images.{cam_suffix}"] = video_arrays[cam][index]
                dataset.add_frame(frame=frame, task=args.task_description)
            dataset.save_episode()
            saved += 1

    logger.info("saved %d episodes to %s", saved, dataset.root)
    meta_dir = Path(dataset.root) / "meta"
    meta_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy(modality_template, meta_dir / "modality.json")
    logger.info("copied modality.json from %s", modality_template)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Convert HDF5 datasets to LeRobot format with multiple cameras.
Reuses functions from IsaacLab-Arena but adds multi-camera support.

Usage:
    python convert_hdf5_to_lerobot.py --config path/to/config.yaml
"""

import argparse
import multiprocessing as mp
import shutil
from pathlib import Path

import h5py
import numpy as np
from isaaclab_arena_gr00t.data_utils.convert_hdf5_to_lerobot import (
    convert_trajectory_to_df,
    generate_info,
    get_video_metadata,
    write_video_job,
)
from isaaclab_arena_gr00t.data_utils.io_utils import create_config_from_yaml, dump_json, dump_jsonl, load_json
from tqdm import tqdm
from utils.assemble_trocar_lerobot_fields import STATE_28_NAMES_ENV_ORDER, convert_g1_state_action_to_lerobot_28d
from utils.extended_dataset_config import ExtendedDatasetConfig


def convert_trajectory_to_df_rheo(
    trajectory,
    episode_index: int,
    index_start: int,
    config: ExtendedDatasetConfig,
) -> dict:
    """
    Convert one rheo trajectory (record_demos_meta_quest.py HDF5) to LeRobot DataFrame.
    """
    obs = trajectory["obs"]
    state_body = np.array(obs["robot_joint_state"])  # (T, 87)
    state_dex3 = np.array(obs["robot_dex3_joint_state"])  # (T, 14)
    state_full = np.concatenate([state_body, state_dex3], axis=1).astype(np.float64)  # (T, 101)
    state_full = state_full[:-1]  # (T-1, 101) align with action

    action_key = getattr(config, "rheo_action_key", "processed_actions")
    action_full = np.array(trajectory[action_key]).astype(np.float64)  # (T, 43)

    use_28d = getattr(config, "rheo_28d_state_action", False)
    if use_28d and state_full.shape[1] >= 29 and action_full.shape[1] >= 43:
        state, action = convert_g1_state_action_to_lerobot_28d(
            state_body=state_body,
            state_dex3=state_dex3,
            action_full=action_full,
        )
    else:
        state = state_full
        action = action_full[:-1]

    length = len(action)
    assert length == len(state), f"state {len(state)} vs action {len(action)}"

    data = {
        config.lerobot_keys["state"]: [state[i] for i in range(length)],
        config.lerobot_keys["action"]: [action[i] for i in range(length)],
        "timestamp": np.arange(length).astype(np.float64) * (1.0 / config.fps),
        "episode_index": np.ones(length, dtype=int) * episode_index,
        "task_index": np.ones(length, dtype=int) * config.task_index,
        "index": np.arange(length, dtype=int) + index_start,
        "frame_index": np.arange(length, dtype=int),
        "next.reward": np.zeros(length, dtype=np.float64),
        "next.done": np.zeros(length, dtype=bool),
    }
    data["next.reward"][-1] = 1.0
    data["next.done"][-1] = True
    data["observation.img_state_delta"] = np.zeros(length, dtype=np.float64)
    if "annotation" in config.lerobot_keys:
        ann_key = config.lerobot_keys["annotation"]
        if isinstance(ann_key, (list, tuple)):
            ann_key = ann_key[0]
        data[ann_key] = np.ones(length, dtype=int) * config.task_index

    dataframe = __import__("pandas").DataFrame(data)
    return {
        "data": dataframe,
        "length": length,
        "annotation": {config.task_index},
    }


def generate_info_rheo(
    total_episodes: int,
    total_frames: int,
    total_tasks: int,
    total_videos: int,
    total_chunks: int,
    config: ExtendedDatasetConfig,
    step_data,
    video_paths: dict,
) -> dict:
    """Build info.json features for rheo; 28-D names aligned with env order (g1_assemble_trocar_env_cfg)."""

    info_template = load_json(config.info_template_path)
    info_template["robot_type"] = config.robot_type
    info_template["total_episodes"] = total_episodes
    info_template["total_frames"] = total_frames
    info_template["total_tasks"] = total_tasks
    info_template["total_videos"] = total_videos
    info_template["total_chunks"] = total_chunks
    info_template["chunks_size"] = config.chunks_size
    info_template["fps"] = config.fps
    info_template["data_path"] = config.data_path
    info_template["video_path"] = config.video_path

    features = {}
    for video_key, video_path in video_paths.items():
        features[video_key] = get_video_metadata(str(video_path))
    state_key = config.lerobot_keys["state"]
    action_key = config.lerobot_keys["action"]
    for column in step_data.columns:
        key = column[0] if isinstance(column, tuple) else column
        column_data = np.stack(step_data[column], axis=0)
        shape = column_data.shape[1:] if column_data.ndim > 1 else (1,)
        features[key] = {"dtype": column_data.dtype.name, "shape": list(shape)}
        if key in (state_key, action_key):
            dof = column_data.shape[1]
            if dof == len(STATE_28_NAMES_ENV_ORDER):
                features[key]["names"] = list(STATE_28_NAMES_ENV_ORDER)
            else:
                features[key]["names"] = [f"dim_{i}" for i in range(dof)]
    info_template["features"] = features
    return info_template


def convert_hdf5_to_lerobot_multivideos(config: ExtendedDatasetConfig):
    """Convert HDF5 to LeRobot format with multi-video support."""

    # Setup video writing workers
    max_queue_size = 10
    num_workers = 4
    queue = mp.Queue(maxsize=max_queue_size)
    error_queue = mp.Queue()

    workers = []
    for _ in range(num_workers):
        worker = mp.Process(target=write_video_job, args=(queue, error_queue, config))
        worker.start()
        workers.append(worker)

    # Load HDF5 file
    hdf5_path = Path(config.hdf5_file_path)
    if not hdf5_path.exists():
        raise FileNotFoundError(f"HDF5 file not found: {hdf5_path}")
    try:
        hdf5_handler = h5py.File(config.hdf5_file_path, "r")
    except OSError as e:
        if "truncated" in str(e).lower() or "eof" in str(e).lower():
            raise OSError(
                f"HDF5 file appears truncated or corrupted: {hdf5_path}\n"
                "The file was likely not fully written (recording interrupted, crash, or disk full).\n"
                "Please use a complete HDF5 from a finished recording or re-record."
            ) from e
        raise
    hdf5_data = hdf5_handler["data"]

    # Create output directories
    config.lerobot_data_dir.mkdir(parents=True, exist_ok=True)
    lerobot_meta_dir = config.lerobot_data_dir / "meta"
    lerobot_meta_dir.mkdir(parents=True, exist_ok=True)

    tasks = {config.task_index: f"{config.language_instruction}"}
    total_length = 0
    example_data = None
    video_paths = {}

    trajectory_ids = list(hdf5_data.keys())
    episodes_info = []

    use_rheo = getattr(config, "use_rheo_converter", False)

    for episode_index, trajectory_id in enumerate(tqdm(trajectory_ids)):
        try:
            trajectory = hdf5_data[trajectory_id]

            if use_rheo:
                df_ret_dict = convert_trajectory_to_df_rheo(
                    trajectory=trajectory, episode_index=episode_index, index_start=total_length, config=config
                )
            else:
                df_ret_dict = convert_trajectory_to_df(
                    trajectory=trajectory, episode_index=episode_index, index_start=total_length, config=config
                )
        except Exception as e:
            raise ValueError(f"Error loading trajectory {trajectory_id}: {e}")

        # Save episode data
        dataframe = df_ret_dict["data"]
        episode_chunk = episode_index // config.chunks_size
        save_relpath = config.data_path.format(episode_chunk=episode_chunk, episode_index=episode_index)
        save_path = config.lerobot_data_dir / save_relpath
        save_path.parent.mkdir(parents=True, exist_ok=True)
        dataframe.to_parquet(save_path)

        # Update metadata
        length = df_ret_dict["length"]
        total_length += length
        episodes_info.append(
            {
                "episode_index": episode_index,
                "tasks": [tasks[task_index] for task_index in df_ret_dict["annotation"]],
                "length": length,
            }
        )

        # === MULTI-CAMERA: Arena camera_obs or rheo obs (front_camera, left_wrist_camera, etc.) ===
        if not use_rheo and config.camera_mappings and "camera_obs" in trajectory.keys():
            for cam_name_sim, video_name_lerobot in config.camera_mappings.items():
                if cam_name_sim not in trajectory["camera_obs"]:
                    continue

                new_video_relpath = config.video_path.format(
                    episode_chunk=episode_chunk, video_key=video_name_lerobot, episode_index=episode_index
                )
                new_video_path = config.lerobot_data_dir / new_video_relpath

                if video_name_lerobot not in video_paths:
                    video_paths[video_name_lerobot] = new_video_path

                frames = np.array(trajectory["camera_obs"][cam_name_sim])
                frames = frames[1:]

                # Convert RGBA to RGB
                if frames.ndim == 4 and frames.shape[-1] == 4:
                    frames = frames[..., :3]

                # Handle single-channel depth images
                if frames.ndim == 4 and frames.shape[-1] == 1:
                    print(
                        f"Processing depth image for {cam_name_sim}: shape={frames.shape}, "
                        f"min={frames.min():.3f}, max={frames.max():.3f}"
                    )

                    # Handle inf/nan values (pixels beyond camera range)
                    valid_mask = np.isfinite(frames)
                    if valid_mask.any():
                        valid_max = frames[valid_mask].max()
                        frames = np.where(np.isfinite(frames), frames, valid_max)

                    # Replace any remaining nan with 0
                    frames = np.nan_to_num(frames, nan=0.0)

                    # Simple normalization to 0-255 range
                    depth_min = frames.min()
                    depth_max = frames.max()

                    if depth_max > depth_min:
                        # Normalize to 0-1 range and convert to uint8
                        frames = (frames - depth_min) / (depth_max - depth_min)
                        frames = (frames * 255).astype(np.uint8)
                    else:
                        frames = np.full_like(frames, 128, dtype=np.uint8)

                    print(f"After normalization: min={frames.min()}, max={frames.max()}, dtype={frames.dtype}")

                    # Convert to 3-channel by repeating
                    frames = np.repeat(frames, 3, axis=-1)

                assert len(frames) == length
                queue.put((new_video_path, frames, config.fps, "image"))

        elif use_rheo and getattr(config, "rheo_camera_mappings_obs", None) and "obs" in trajectory:
            for hdf5_key, video_name_lerobot in config.rheo_camera_mappings_obs.items():
                if hdf5_key not in trajectory["obs"]:
                    continue
                new_video_relpath = config.video_path.format(
                    episode_chunk=episode_chunk, video_key=video_name_lerobot, episode_index=episode_index
                )
                new_video_path = config.lerobot_data_dir / new_video_relpath
                if video_name_lerobot not in video_paths:
                    video_paths[video_name_lerobot] = new_video_path
                frames = np.array(trajectory["obs"][hdf5_key])
                frames = frames[1:]
                if frames.ndim == 4 and frames.shape[-1] == 4:
                    frames = frames[..., :3]
                assert len(frames) == length
                queue.put((new_video_path, frames, config.fps, "image"))

        if example_data is None:
            example_data = df_ret_dict

    # Generate metadata files
    tasks_path = lerobot_meta_dir / config.tasks_fname
    task_jsonlines = [{"task_index": task_index, "task": task} for task_index, task in tasks.items()]
    dump_jsonl(task_jsonlines, tasks_path)

    episodes_path = lerobot_meta_dir / config.episodes_fname
    dump_jsonl(episodes_info, episodes_path)

    modality_path = lerobot_meta_dir / config.modality_fname
    shutil.copy(config.modality_template_path, modality_path)

    # Signal workers to finish
    for _ in range(num_workers):
        queue.put(None)

    for worker in workers:
        worker.join()

    # Generate info.json (after all videos are created)
    if use_rheo:
        info_json = generate_info_rheo(
            total_episodes=len(trajectory_ids),
            total_frames=total_length,
            total_tasks=len(tasks),
            total_videos=len(video_paths) or len(trajectory_ids),
            total_chunks=max(1, len(trajectory_ids) // config.chunks_size),
            config=config,
            step_data=example_data["data"],
            video_paths=video_paths,
        )
    else:
        info_json = generate_info(
            total_episodes=len(trajectory_ids),
            total_frames=total_length,
            total_tasks=len(tasks),
            total_videos=len(trajectory_ids),
            total_chunks=len(trajectory_ids) // config.chunks_size,
            step_data=example_data["data"],
            video_paths=video_paths,
            config=config,
        )
    dump_json(info_json, lerobot_meta_dir / "info.json", indent=4)

    hdf5_handler.close()


def main():
    parser = argparse.ArgumentParser(description="Convert HDF5 to LeRobot with multi-camera support")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config file")
    args = parser.parse_args()

    config = create_config_from_yaml(args.config, ExtendedDatasetConfig)
    convert_hdf5_to_lerobot_multivideos(config)


if __name__ == "__main__":
    main()

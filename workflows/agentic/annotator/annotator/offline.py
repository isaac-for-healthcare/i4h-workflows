# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import logging
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import h5py
import numpy as np
from annotator.records import EpisodeSample, FrameBundle

logger = logging.getLogger("annotator.offline")


def iter_hdf5_episodes(
    *,
    hdf5_path: str,
    cameras: list[str] | None,
    sample_frames: int,
    max_episodes: int | None,
) -> Iterator[EpisodeSample]:
    path = Path(hdf5_path).expanduser()
    if not path.exists():
        raise FileNotFoundError(path)

    with h5py.File(path, "r") as file:
        for count, (episode_id, group) in enumerate(_open_demos(file)):
            if max_episodes is not None and count >= max_episodes:
                break
            camera_keys = _camera_keys(group, cameras)
            if not camera_keys:
                logger.info("skip %s (no camera arrays)", episode_id)
                continue
            total_frames = min(group[f"obs/{camera}"].shape[0] for camera in camera_keys)
            if total_frames <= 0:
                logger.info("skip %s (empty cameras)", episode_id)
                continue
            yield EpisodeSample(
                episode_id=episode_id,
                total_frames=total_frames,
                frames=[
                    FrameBundle(
                        index=index,
                        cameras={camera: np.asarray(group[f"obs/{camera}"][index]) for camera in camera_keys},
                    )
                    for index in _sample_indices(total_frames, sample_frames)
                ],
            )


def write_successful_episodes(
    *,
    source_hdf5: str,
    output_hdf5: str,
    records: list[dict[str, Any]],
) -> tuple[int, int]:
    """Write a filtered HDF5 with only successfully annotated offline episodes."""
    successful = [record for record in records if bool(record.get("annotation", {}).get("success", False))]
    if not successful:
        raise ValueError("no successful annotated episodes; refusing to write filtered HDF5")

    source_path = Path(source_hdf5).expanduser()
    output_path = Path(output_hdf5).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(source_path, "r") as source, h5py.File(output_path, "w") as output:
        _copy_attrs(source.attrs, output.attrs)
        source_data = source["data"]
        output_data = output.create_group("data")
        _copy_attrs(source_data.attrs, output_data.attrs)

        for index, record in enumerate(successful):
            episode_id = record.get("episode_id")
            if not episode_id:
                raise ValueError("successful annotation record is missing episode_id")
            if episode_id not in source_data:
                raise KeyError(f"annotation references missing episode {episode_id!r}")

            target_name = f"demo_{index}"
            source.copy(source_data[episode_id], output_data, name=target_name)
            output_data[target_name].attrs["source_demo"] = episode_id
            output_data[target_name].attrs["annotation_success"] = True
            output_data[target_name].attrs["annotation_reasoning"] = record.get("annotation", {}).get("reasoning", "")

        output_data.attrs["total"] = len(successful)
        output.attrs["filtered_from_hdf5"] = str(source_path)

    return len(successful), len(records) - len(successful)


def _open_demos(file: h5py.File) -> list[tuple[str, h5py.Group]]:
    root = file.get("data")
    if root is None:
        raise ValueError("HDF5 recording must contain a data/ group")
    return [(name, group) for name, group in root.items() if hasattr(group, "keys")]


def _copy_attrs(source: h5py.AttributeManager, target: h5py.AttributeManager) -> None:
    for key, value in source.items():
        target[key] = value


def _camera_keys(group: h5py.Group, override: list[str] | None) -> list[str]:
    if override:
        missing = [camera for camera in override if f"obs/{camera}" not in group]
        if missing:
            raise KeyError(f"episode {group.name} missing camera arrays: {missing}")
        return list(override)
    obs = group.get("obs")
    if obs is None:
        return []
    return [name for name, value in obs.items() if _is_rgb_video_dataset(value)]


def _is_rgb_video_dataset(value: Any) -> bool:
    return (
        hasattr(value, "shape")
        and hasattr(value, "dtype")
        and value.ndim == 4
        and value.shape[-1] == 3
        and value.dtype == np.uint8
    )


def _sample_indices(total_frames: int, sample_frames: int) -> list[int]:
    count = max(1, min(sample_frames, total_frames))
    return sorted({int(round(value)) for value in np.linspace(0, total_frames - 1, num=count)})

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py


@dataclass(frozen=True)
class DemoSummary:
    name: str
    length: int
    action_shape: tuple[int, ...]
    state_shape: tuple[int, ...] | None
    cameras: dict[str, tuple[int, ...]]
    success: bool | None


def data_group(file: h5py.File) -> h5py.Group:
    if "data" not in file or not isinstance(file["data"], h5py.Group):
        raise KeyError("expected root group 'data'")
    return file["data"]


def demo_names(file_or_group: h5py.File | h5py.Group) -> list[str]:
    root = file_or_group["data"] if isinstance(file_or_group, h5py.File) and "data" in file_or_group else file_or_group
    return sorted(
        (name for name in root.keys() if name.startswith("demo_") and _has_action_dataset(root[name])),
        key=_demo_sort_key,
    )


def action_dataset_path(demo: h5py.Group) -> str:
    if "obs/actions" in demo:
        return "obs/actions"
    if "actions" in demo:
        return "actions"
    raise KeyError(f"{demo.name}: expected obs/actions or actions")


def _has_action_dataset(demo: h5py.Group) -> bool:
    return "obs/actions" in demo or "actions" in demo


def state_dataset_path(demo: h5py.Group) -> str | None:
    return "obs/joint_pos" if "obs/joint_pos" in demo else None


def camera_keys(demo: h5py.Group) -> list[str]:
    obs = demo.get("obs")
    if obs is None:
        return []
    return [
        name
        for name, dataset in obs.items()
        if hasattr(dataset, "shape") and dataset.ndim == 4 and dataset.shape[-1] == 3
    ]


def summarize_demo(name: str, demo: h5py.Group) -> DemoSummary:
    action_path = action_dataset_path(demo)
    state_path = state_dataset_path(demo)
    success = demo.attrs.get("success")
    return DemoSummary(
        name=name,
        length=int(demo[action_path].shape[0]),
        action_shape=tuple(demo[action_path].shape),
        state_shape=tuple(demo[state_path].shape) if state_path else None,
        cameras={camera: tuple(demo[f"obs/{camera}"].shape) for camera in camera_keys(demo)},
        success=bool(success) if success is not None else None,
    )


def validate_demo(demo: h5py.Group) -> None:
    action_path = action_dataset_path(demo)
    n = int(demo[action_path].shape[0])
    if n <= 0:
        raise ValueError(f"{demo.name}: actions dataset is empty")
    state_path = state_dataset_path(demo)
    if state_path and int(demo[state_path].shape[0]) != n:
        raise ValueError(f"{demo.name}: {state_path} length does not match actions")


def copy_attrs(src: h5py.AttributeManager, dst: h5py.AttributeManager) -> None:
    for key, value in src.items():
        dst[key] = value


def copy_item(src: h5py.Group | h5py.Dataset, dst_parent: h5py.Group, name: str) -> None:
    if isinstance(src, h5py.Dataset):
        dst = dst_parent.create_dataset(name, data=src[...], **dataset_kwargs(src))
        copy_attrs(src.attrs, dst.attrs)
        return
    dst = dst_parent.create_group(name)
    copy_attrs(src.attrs, dst.attrs)
    for child_name, child in src.items():
        copy_item(child, dst, child_name)


def copy_demo(src_demo: h5py.Group, dst_data: h5py.Group, dst_name: str) -> h5py.Group:
    dst_demo = dst_data.create_group(dst_name)
    copy_attrs(src_demo.attrs, dst_demo.attrs)
    for child_name, child in src_demo.items():
        copy_item(child, dst_demo, child_name)
    return dst_demo


def copy_root_metadata(src_file: h5py.File, dst_file: h5py.File) -> h5py.Group:
    copy_attrs(src_file.attrs, dst_file.attrs)
    dst_data = dst_file.create_group("data")
    if "data" in src_file:
        copy_attrs(src_file["data"].attrs, dst_data.attrs)
    return dst_data


def dataset_kwargs(dataset: h5py.Dataset) -> dict:
    kwargs = {}
    for attr in ("compression", "compression_opts", "shuffle", "fletcher32"):
        value = getattr(dataset, attr)
        if value:
            kwargs[attr] = value
    return kwargs


def set_total(data: h5py.Group, total: int) -> None:
    data.attrs["total"] = int(total)


def inspect_file(path: str | Path) -> list[DemoSummary]:
    with h5py.File(path, "r") as file:
        root = data_group(file)
        return [summarize_demo(name, root[name]) for name in demo_names(root)]


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("_", 1)[1]), name)
    except (IndexError, ValueError):
        return (10**9, name)

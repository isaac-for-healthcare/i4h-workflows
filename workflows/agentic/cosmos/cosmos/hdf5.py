# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

from pathlib import Path

import h5py


def data_group(file: h5py.File) -> h5py.Group:
    if "data" not in file or not isinstance(file["data"], h5py.Group):
        raise KeyError("expected root group 'data'")
    return file["data"]


def demo_names(file_or_group: h5py.File | h5py.Group) -> list[str]:
    root = file_or_group["data"] if isinstance(file_or_group, h5py.File) and "data" in file_or_group else file_or_group
    return sorted((name for name in root.keys() if name.startswith("demo_")), key=_demo_sort_key)


def camera_keys(demo: h5py.Group) -> tuple[str, ...]:
    obs = demo.get("obs")
    if obs is None:
        return ()
    return tuple(
        name
        for name, dataset in obs.items()
        if hasattr(dataset, "shape") and dataset.ndim == 4 and dataset.shape[-1] == 3
    )


def action_dataset_path(demo: h5py.Group) -> str:
    if "obs/actions" in demo:
        return "obs/actions"
    if "actions" in demo:
        return "actions"
    raise KeyError(f"{demo.name}: expected obs/actions or actions")


def validate_demo(demo: h5py.Group, cameras: tuple[str, ...]) -> int:
    action_path = action_dataset_path(demo)
    length = int(demo[action_path].shape[0])
    if "obs/joint_pos" in demo and int(demo["obs/joint_pos"].shape[0]) != length:
        raise ValueError(f"{demo.name}: obs/joint_pos length does not match actions")
    for camera in cameras:
        key = f"obs/{camera}"
        if key not in demo:
            raise KeyError(f"{demo.name}: expected {key}")
        if int(demo[key].shape[0]) != length:
            raise ValueError(f"{demo.name}: {key} length does not match actions")
    return length


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


def replace_dataset(parent: h5py.Group, name: str, source_dataset: h5py.Dataset, data) -> h5py.Dataset:
    if name in parent:
        del parent[name]
    dst = parent.create_dataset(name, data=data, **dataset_kwargs(source_dataset))
    copy_attrs(source_dataset.attrs, dst.attrs)
    return dst


def set_total(data: h5py.Group, total: int) -> None:
    data.attrs["total"] = int(total)


def ensure_parent(path: str | Path) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)


def _demo_sort_key(name: str) -> tuple[int, str]:
    try:
        return (int(name.split("_", 1)[1]), name)
    except (IndexError, ValueError):
        return (10**9, name)

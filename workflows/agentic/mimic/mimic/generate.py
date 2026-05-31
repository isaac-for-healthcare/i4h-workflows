# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
from common.config import get_environment_config, get_robot_config
from common.utils import resolve_path
from mimic.hdf5 import (
    action_dataset_path,
    copy_demo,
    copy_root_metadata,
    data_group,
    demo_names,
    set_total,
    state_dataset_path,
    validate_demo,
)
from tqdm import tqdm


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Generate Agentic HDF5 demo variants.")
    parser.add_argument("--input", required=True, help="Input HDF5 file.")
    parser.add_argument("--output", required=True, help="Generated HDF5 output path.")
    parser.add_argument("--episodes", type=int, required=True, help="Number of generated episodes to write.")
    parser.add_argument(
        "--env", default=None, help="Environment id from config/environments/<env>.yaml for joint limits."
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed for generation.")
    parser.add_argument("--noise-std", type=float, default=0.01, help="Gaussian action/state noise.")
    parser.add_argument("--include-source", action="store_true", help="Copy source demos before generated variants.")
    parser.add_argument("--overwrite", action="store_true", help="Remove the output file before generation.")
    return parser


def main() -> None:
    args = build_argparser().parse_args()
    input_path = resolve_path(args.input, args.env)
    output_path = resolve_path(args.output, args.env)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    if args.episodes < 1:
        raise ValueError("--episodes must be >= 1")
    if args.noise_std < 0:
        raise ValueError("--noise-std must be >= 0")
    _prepare_output(output_path, overwrite=args.overwrite)

    written = generate_simple(
        input_path=input_path,
        output_path=output_path,
        episodes=args.episodes,
        seed=args.seed,
        noise_std=args.noise_std,
        env=args.env,
        include_source=args.include_source,
    )
    print(f"wrote {written} demos with simple mimic: {output_path}", flush=True)


def generate_simple(
    input_path: Path,
    output_path: Path,
    episodes: int,
    seed: int,
    noise_std: float,
    env: str | None = None,
    include_source: bool = False,
) -> int:
    rng = np.random.default_rng(seed)
    joint_limits = _joint_limits_rad(env)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        src_data = data_group(src)
        seeds = demo_names(src_data)
        if not seeds:
            raise ValueError(f"no demo_* groups found in {input_path}")

        dst_data = copy_root_metadata(src, dst)
        written = 0
        if include_source:
            for source_name in seeds:
                validate_demo(src_data[source_name])
                copy_demo(src_data[source_name], dst_data, f"demo_{written}")
                written += 1

        schedule = tqdm(_source_schedule(seeds, episodes, rng), desc="generated demos")
        for variant_index, source_name in enumerate(schedule):
            src_demo = src_data[source_name]
            validate_demo(src_demo)
            dst_demo = copy_demo(src_demo, dst_data, f"demo_{written}")
            _jitter_demo(dst_demo, rng, noise_std, joint_limits)
            dst_demo.attrs["source_demo"] = source_name
            dst_demo.attrs["variant_index"] = variant_index
            dst_demo.attrs["mimic_backend"] = "simple"
            dst_demo.attrs.setdefault("success", True)
            written += 1

        set_total(dst_data, written)
        dst.attrs["mimic_generated_from"] = str(input_path)
        dst.attrs["mimic_backend"] = "simple"
        dst.attrs["mimic_noise_std"] = float(noise_std)
    return written


def _source_schedule(seeds: list[str], episodes: int, rng: np.random.Generator) -> list[str]:
    source_order = list(seeds)
    rng.shuffle(source_order)
    source_count = len(source_order)
    return [source_order[(index + index // source_count) % source_count] for index in range(episodes)]


def _jitter_demo(
    demo: h5py.Group,
    rng: np.random.Generator,
    noise_std: float,
    joint_limits: np.ndarray | None,
) -> None:
    _jitter_dataset(demo[action_dataset_path(demo)], rng, noise_std, joint_limits)
    state_path = state_dataset_path(demo)
    if state_path:
        _jitter_dataset(demo[state_path], rng, noise_std, joint_limits)


def _jitter_dataset(
    dataset: h5py.Dataset,
    rng: np.random.Generator,
    noise_std: float,
    joint_limits: np.ndarray | None,
) -> None:
    if noise_std == 0:
        return
    values = dataset[...]
    if not np.issubdtype(values.dtype, np.number):
        return
    working = values.astype(np.float32, copy=True)
    noise = rng.normal(0.0, noise_std, size=working.shape).astype(np.float32)
    if noise.ndim >= 2 and noise.shape[0] > 0:
        noise[0] = 0.0
    working += noise
    if joint_limits is not None and working.shape[-1] == len(joint_limits):
        working = np.clip(working, joint_limits[:, 0], joint_limits[:, 1])
    dataset[...] = working.astype(values.dtype, copy=False)


def _joint_limits_rad(env: str | None) -> np.ndarray | None:
    if not env:
        return None
    env_config = get_environment_config(env)
    dataset = env_config.get("dataset", {})
    robot_type = dataset.get("robot_type") or (env_config.get("robot") or {}).get("type")
    limits = dataset.get("joint_source_range") or _default_joint_source_range(robot_type)
    if not limits:
        return None
    limits = np.asarray(limits, dtype=np.float32)
    return np.deg2rad(limits) if dataset.get("joint_source_unit") == "rad" else limits


def _default_joint_source_range(robot_type: str | None):
    if robot_type:
        return get_robot_config(robot_type).isaaclab_joint_pos_limit_range
    return None


def _prepare_output(output_path: Path, overwrite: bool) -> None:
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"{output_path} already exists; pass --overwrite to replace it")
    if output_path.exists():
        output_path.unlink()
    output_path.parent.mkdir(parents=True, exist_ok=True)


if __name__ == "__main__":
    main()

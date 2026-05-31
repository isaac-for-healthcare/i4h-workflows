# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import h5py
from common.utils import resolve_path
from cosmos.hdf5 import copy_demo, copy_root_metadata, data_group, demo_names, replace_dataset, set_total
from cosmos.video import read_video
from tqdm import tqdm


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Import Cosmos-generated videos into an expanded Agentic HDF5.")
    parser.add_argument("--env", required=True, help="Environment id for resolving relative paths under runs/<env>.")
    parser.add_argument("--manifest", required=True, help="Manifest produced by agentic-cosmos-export.")
    parser.add_argument("--output", required=True, help="Expanded HDF5 output.")
    parser.add_argument("--include-original", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--use-source-videos", action="store_true", help="Use source videos when Cosmos outputs are missing."
    )
    return parser


def import_generated(
    manifest_path: Path,
    output_path: Path,
    include_original: bool = True,
    use_source_videos: bool = False,
) -> int:
    manifest = json.loads(manifest_path.read_text())
    input_path = Path(manifest["input_hdf5"])
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(input_path, "r") as src, h5py.File(output_path, "w") as dst:
        src_data = data_group(src)
        dst_data = copy_root_metadata(src, dst)
        written = 0
        if include_original:
            for source_demo in demo_names(src_data):
                copy_demo(src_data[source_demo], dst_data, f"demo_{written}")
                written += 1

        for demo_entry in tqdm(manifest["demos"], desc="import variants"):
            source_demo_name = demo_entry["name"]
            source_demo = src_data[source_demo_name]
            for variant_index in range(int(manifest["variants"])):
                dst_demo = copy_demo(source_demo, dst_data, f"demo_{written}")
                _replace_cameras(dst_demo, source_demo, demo_entry, variant_index, use_source_videos)
                dst_demo.attrs["source_demo"] = source_demo_name
                dst_demo.attrs["cosmos_variant"] = variant_index
                dst_demo.attrs["cosmos_prompt"] = manifest["prompt"]
                dst_demo.attrs.setdefault("success", True)
                written += 1
        set_total(dst_data, written)
        dst.attrs["cosmos_manifest"] = str(manifest_path)
        dst.attrs["cosmos_source_hdf5"] = str(input_path)
    return written


def missing_generated_outputs(manifest_path: Path) -> list[str]:
    manifest = json.loads(manifest_path.read_text())
    missing = []
    for demo_entry in manifest["demos"]:
        for variant_index in range(int(manifest["variants"])):
            for camera, camera_entry in demo_entry["cameras"].items():
                if _resolve_generated_video(camera_entry, variant_index) is None:
                    missing.append(f"{demo_entry['name']}/{camera}/variant_{variant_index:03d}")
    return missing


def _replace_cameras(
    dst_demo, source_demo, demo_entry: dict[str, Any], variant_index: int, use_source_videos: bool
) -> None:
    for camera, camera_entry in demo_entry["cameras"].items():
        generated_video = _resolve_generated_video(camera_entry, variant_index)
        if generated_video is None:
            if not use_source_videos:
                raise FileNotFoundError(
                    f"missing Cosmos output for {demo_entry['name']}/{camera}/variant_{variant_index:03d}"
                )
            generated_video = Path(camera_entry["source_video"])
        frames = read_video(generated_video)
        source_dataset = source_demo[f"obs/{camera}"]
        if tuple(frames.shape) != tuple(source_dataset.shape):
            raise ValueError(
                f"{generated_video}: generated frames shape {frames.shape} does not match {source_dataset.shape}"
            )
        replace_dataset(dst_demo["obs"], camera, source_dataset, frames)


def _resolve_generated_video(camera_entry: dict[str, Any], variant_index: int) -> Path | None:
    variant = next(entry for entry in camera_entry["variants"] if int(entry["variant"]) == variant_index)
    output_dir = Path(variant["output_dir"])
    if not output_dir.exists():
        return None
    videos = sorted(output_dir.glob("*.mp4")) or sorted(output_dir.glob("**/*.mp4"))
    return videos[0] if videos else None


def main() -> None:
    args = build_argparser().parse_args()
    manifest_path = resolve_path(args.manifest, args.env)
    if not manifest_path.exists():
        raise FileNotFoundError(manifest_path)
    output_path = resolve_path(args.output, args.env)
    written = import_generated(
        manifest_path=manifest_path,
        output_path=output_path,
        include_original=args.include_original,
        use_source_videos=args.use_source_videos,
    )
    print(f"wrote {written} demos: {output_path}")


if __name__ == "__main__":
    main()

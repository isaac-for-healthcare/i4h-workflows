# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
from common.utils import resolve_path
from cosmos.hdf5 import camera_keys, data_group, demo_names, validate_demo
from cosmos.video import write_video
from tqdm import tqdm


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Export Agentic HDF5 camera streams to Cosmos Transfer specs.")
    parser.add_argument("--env", required=True, help="Environment id for resolving relative paths under runs/<env>.")
    parser.add_argument("--input", required=True, help="Input HDF5.")
    parser.add_argument("--workspace", required=True, help="Workspace for exported videos, specs, and manifest.")
    parser.add_argument("--variants", type=int, required=True, help="Number of Cosmos variants per source demo.")
    parser.add_argument("--prompt", required=True, help="Cosmos text prompt.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--control", choices=("vis", "edge"), default="vis")
    parser.add_argument("--camera", action="append", help="Camera key to export. Defaults to auto-detected cameras.")
    return parser


def export_specs(
    input_path: Path,
    workspace: Path,
    variants: int,
    prompt: str,
    fps: int = 30,
    guidance: float = 3.0,
    control: str = "vis",
    cameras: tuple[str, ...] | None = None,
) -> Path:
    if variants < 1:
        raise ValueError("--variants must be >= 1")
    workspace.mkdir(parents=True, exist_ok=True)
    manifest = _manifest(input_path, workspace, variants, prompt, fps, guidance, control, cameras)

    source_video_dir = workspace / "source_videos"
    specs_dir = workspace / "specs"
    outputs_dir = workspace / "outputs"
    with h5py.File(input_path, "r") as file:
        root = data_group(file)
        for demo_name in tqdm(demo_names(root), desc="export demos"):
            demo = root[demo_name]
            selected_cameras = cameras or camera_keys(demo)
            length = validate_demo(demo, cameras=selected_cameras)
            demo_entry: dict[str, object] = {"name": demo_name, "length": length, "cameras": {}}
            for camera in selected_cameras:
                demo_entry["cameras"][camera] = _export_camera(
                    demo,
                    demo_name,
                    camera,
                    source_video_dir,
                    specs_dir,
                    outputs_dir,
                    variants,
                    prompt,
                    fps,
                    guidance,
                    control,
                )
            manifest["demos"].append(demo_entry)

    manifest_path = workspace / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest_path


def _manifest(input_path, workspace, variants, prompt, fps, guidance, control, cameras) -> dict[str, object]:
    return {
        "workflow": "agentic-cosmos-transfer",
        "input_hdf5": str(input_path.resolve()),
        "workspace": str(workspace.resolve()),
        "variants": variants,
        "fps": fps,
        "prompt": prompt,
        "guidance": guidance,
        "control": control,
        "cameras": list(cameras or []),
        "demos": [],
    }


def _export_camera(
    demo,
    demo_name: str,
    camera: str,
    source_video_dir: Path,
    specs_dir: Path,
    outputs_dir: Path,
    variants: int,
    prompt: str,
    fps: int,
    guidance: float,
    control: str,
) -> dict[str, object]:
    source_video = source_video_dir / demo_name / f"{camera}.mp4"
    write_video(source_video, demo[f"obs/{camera}"][...], fps=fps)
    camera_entry: dict[str, object] = {"source_video": str(source_video.resolve()), "variants": []}
    for variant_index in range(variants):
        variant_name = f"variant_{variant_index:03d}"
        output_dir = outputs_dir / demo_name / variant_name / camera
        params_file = specs_dir / demo_name / variant_name / f"{camera}.json"
        output_dir.mkdir(parents=True, exist_ok=True)
        params_file.parent.mkdir(parents=True, exist_ok=True)
        params_file.write_text(
            json.dumps(
                _cosmos_params(
                    f"{demo_name}_{variant_name}_{camera}",
                    prompt,
                    source_video.resolve(),
                    guidance,
                    variant_index,
                    control,
                ),
                indent=2,
            )
            + "\n"
        )
        camera_entry["variants"].append(
            {
                "variant": variant_index,
                "params_file": str(params_file.resolve()),
                "output_dir": str(output_dir.resolve()),
            }
        )
    return camera_entry


def _cosmos_params(name: str, prompt: str, video_path: Path, guidance: float, seed: int, control: str) -> dict:
    params: dict[str, object] = {
        "name": name,
        "prompt": prompt,
        "video_path": str(video_path),
        "guidance": guidance,
        "seed": seed,
    }
    params[control] = {"control_weight": 0.5}
    return params


def main() -> None:
    args = build_argparser().parse_args()
    input_path = resolve_path(args.input, args.env)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    manifest_path = export_specs(
        input_path=input_path,
        workspace=resolve_path(args.workspace, args.env),
        variants=args.variants,
        prompt=args.prompt,
        fps=args.fps,
        guidance=args.guidance,
        control=args.control,
        cameras=tuple(args.camera) if args.camera else None,
    )
    print(f"manifest: {manifest_path}")


if __name__ == "__main__":
    main()

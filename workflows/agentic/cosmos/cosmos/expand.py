# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import os
import subprocess
from pathlib import Path

from common.utils import resolve_path
from cosmos.export import export_specs
from cosmos.importer import import_generated, missing_generated_outputs
from cosmos.run import command_for_params, iter_params_files, validate_cosmos_runtime


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Expand Agentic HDF5 episodes with Cosmos visual variants.")
    parser.add_argument("--env", required=True, help="Environment id for resolving relative paths under runs/<env>.")
    parser.add_argument("--input", required=True, help="Input HDF5.")
    parser.add_argument("--output", required=True, help="Expanded HDF5 output.")
    parser.add_argument("--variants", type=int, required=True, help="Number of visual variants per source demo.")
    parser.add_argument("--prompt", required=True, help="Cosmos Transfer prompt.")
    parser.add_argument("--workspace", default=None, help="Working directory for exported videos/specs.")
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--guidance", type=float, default=3.0)
    parser.add_argument("--control", choices=("vis", "edge"), default="vis")
    parser.add_argument("--camera", action="append", help="Camera key to expand. Defaults to auto-detected cameras.")
    parser.add_argument("--run-cosmos", action="store_true", help="Run Cosmos through Docker after exporting specs.")
    parser.add_argument("--docker-image", default="cosmos-transfer-2.5", help="Docker image for --run-cosmos.")
    parser.add_argument("--cosmos-root", default=os.environ.get("COSMOS_TRANSFER_ROOT"))
    parser.add_argument(
        "--use-source-videos", action="store_true", help="Use source videos if Cosmos outputs are missing."
    )
    parser.add_argument("--no-include-original", action="store_true", help="Write only generated variants.")
    return parser


def expand(
    input_path: Path,
    output_path: Path,
    variants: int,
    prompt: str,
    workspace: Path,
    fps: int = 30,
    guidance: float = 3.0,
    control: str = "vis",
    cameras: tuple[str, ...] | None = None,
    run_cosmos: bool = False,
    docker_image: str = "cosmos-transfer-2.5",
    cosmos_root: Path | None = None,
    use_source_videos: bool = False,
    include_original: bool = True,
) -> tuple[Path, bool]:
    manifest_path = export_specs(input_path, workspace, variants, prompt, fps, guidance, control, cameras)
    if run_cosmos:
        validate_cosmos_runtime(docker_image, cosmos_root)
        for job in iter_params_files(manifest_path):
            subprocess.run(command_for_params(job.params_file, job.output_dir, docker_image, cosmos_root), check=True)

    if missing_generated_outputs(manifest_path) and not use_source_videos:
        return manifest_path, False
    import_generated(manifest_path, output_path, include_original, use_source_videos)
    return manifest_path, True


def main() -> None:
    args = build_argparser().parse_args()
    input_path = resolve_path(args.input, args.env)
    if not input_path.exists():
        raise FileNotFoundError(input_path)
    output_path = resolve_path(args.output, args.env)
    cosmos_root = Path(args.cosmos_root) if args.cosmos_root else None
    if cosmos_root is not None and not cosmos_root.exists():
        raise FileNotFoundError(cosmos_root)
    workspace = (
        resolve_path(args.workspace, args.env)
        if args.workspace
        else output_path.parent / f"{output_path.stem}_cosmos_workdir"
    )
    manifest_path, imported = expand(
        input_path=input_path,
        output_path=output_path,
        variants=args.variants,
        prompt=args.prompt,
        workspace=workspace,
        fps=args.fps,
        guidance=args.guidance,
        control=args.control,
        cameras=tuple(args.camera) if args.camera else None,
        run_cosmos=args.run_cosmos,
        docker_image=args.docker_image,
        cosmos_root=cosmos_root,
        use_source_videos=args.use_source_videos,
        include_original=not args.no_include_original,
    )
    print(f"manifest: {manifest_path}")
    print(f"expanded: {output_path}" if imported else "Cosmos outputs are not available yet; no HDF5 was written.")


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Export recording videos for visual augmentation and import them afterward.

The adapter deliberately leaves model execution outside this dependency-light
project. ``export`` writes each selected camera stream as MP4. After an
augmentation service writes replacement clips, ``import`` copies the complete
source demo and replaces only that camera. Actions, states, attributes, other
cameras, and workflow segments remain unchanged.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from i4h_common.episode import DATA_GROUP, Episode, camera_keys, demo_names, read_segments

MANIFEST = "manifest.json"

logger = logging.getLogger("i4h_tools.cosmos")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-cosmos", description="Cosmos visual expansion adapter.")
    sub = parser.add_subparsers(dest="command", required=True)

    export_cmd = sub.add_parser("export", help="recording → MP4 per demo per camera")
    export_cmd.add_argument("input", type=Path)
    export_cmd.add_argument("outdir", type=Path)
    export_cmd.add_argument("--camera", default=None, help="defaults to every camera present")
    export_cmd.add_argument("--fps", type=int, default=30)
    export_cmd.add_argument("--node", default=None, help="export only this workflow node's frames")

    import_cmd = sub.add_parser("import", help="augmented MP4s → new demos in a recording")
    import_cmd.add_argument("input", type=Path, help="source recording (for actions and segments)")
    import_cmd.add_argument("videodir", type=Path, help="directory of augmented MP4s")
    import_cmd.add_argument("output", type=Path)
    import_cmd.add_argument("--camera", required=True)
    import_cmd.add_argument("--include-source", action="store_true")
    return parser


def export(source: Path, outdir: Path, *, camera: str | None = None, fps: int = 30, node: str | None = None) -> int:
    import imageio.v3 as iio  # noqa: PLC0415

    outdir.mkdir(parents=True, exist_ok=True)
    written = 0
    with h5py.File(str(source), "r") as handle:
        data = handle[DATA_GROUP]
        for name in tqdm(demo_names(handle), desc="export"):
            demo = data[name]
            cameras = [camera] if camera else camera_keys(demo)
            for key in cameras:
                if key not in camera_keys(demo):
                    continue
                video = demo[f"obs/{key}"][()]
                if node:
                    segment = next((s for s in read_segments(demo) if s.node == node), None)
                    if segment is None:
                        continue
                    video = video[segment.start : segment.end]
                path = outdir / f"{name}__{key}.mp4"
                iio.imwrite(str(path), video.astype(np.uint8), fps=fps, codec="libx264")
                written += 1
    logger.info("exported %s clips to %s", written, outdir)
    return written


def import_videos(source: Path, videodir: Path, target: Path, *, camera: str, include_source: bool = False) -> int:
    """Fold augmented videos back in, reusing each source demo's actions."""
    import imageio.v3 as iio  # noqa: PLC0415

    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(source), "r") as src, h5py.File(str(target), "w") as dst:
        src_data = src[DATA_GROUP]
        dst_data = dst.create_group(DATA_GROUP)
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value

        written = 0
        if include_source:
            for name in demo_names(src):
                _copy(src_data[name], dst_data, f"demo_{written}")
                written += 1

        for path in tqdm(sorted(videodir.glob("*.mp4")), desc="import"):
            origin = path.stem.split("__")[0]
            if origin not in src_data:
                logger.warning("%s has no matching demo %s in the source; skipping", path.name, origin)
                continue
            frames = np.asarray(iio.imread(str(path)), dtype=np.uint8)
            source_demo = src_data[origin]
            episode = Episode(origin, source_demo)
            actions = episode.actions
            if len(frames) != len(actions):
                raise ValueError(
                    f"{path}: {len(frames)} frames does not match {len(actions)} source actions; "
                    "resample the video before importing it"
                )

            demo = _copy(source_demo, dst_data, f"demo_{written}")
            obs = demo.require_group("obs")
            if camera in obs:
                del obs[camera]
            obs.create_dataset(camera, data=frames, compression="gzip", compression_opts=4)
            demo.attrs["cosmos_source"] = origin
            demo.attrs["cosmos_video"] = path.name
            written += 1

        dst_data.attrs["total"] = written
        logger.info("wrote %s demos to %s", written, target)
        return written


def _copy(src: h5py.Group, dst_parent: h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            _copy(item, dst, key)
        else:
            copied = dst.create_dataset(key, data=item[()], dtype=item.dtype)
            for attr, value in item.attrs.items():
                copied.attrs[attr] = value
    return dst


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[cosmos] %(message)s")
    if not args.input.is_file():
        print(f"error: no recording at {args.input}")
        return 1
    if args.command == "export":
        count = export(args.input, args.outdir, camera=args.camera, fps=args.fps, node=args.node)
        print(f"exported {count} clips to {args.outdir}")
        return 0
    count = import_videos(
        args.input, args.videodir, args.output, camera=args.camera, include_source=args.include_source
    )
    print(f"wrote {count} demos to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

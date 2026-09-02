# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Expand a recording by cloning demos with action/state noise.

Offline artifact processing: file in, file out. No ``engine`` dependency,
because these are not workflow nodes.

Recordings carry per-node segments (see :mod:`i4h_common.episode`), so noise
can be applied to a *single skill* — jitter the grasp, leave the approach and
the carry untouched. Augmenting a whole episode uniformly is the blunt version
of that, and it is what makes half the generated demos useless.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from i4h_common.episode import DATA_GROUP, action_path, demo_names, read_segments, write_segments

logger = logging.getLogger("i4h_tools.mimic")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-mimic", description="Clone HDF5 demos with noise.")
    parser.add_argument("input", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--episodes", type=int, default=10, help="how many variants to generate")
    parser.add_argument("--noise", type=float, default=0.01, help="std-dev of the action jitter, in action units")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--node",
        default=None,
        help="only jitter frames recorded while this workflow node was active (needs a node-tagged recording)",
    )
    parser.add_argument("--include-source", action="store_true", help="copy the source demos in first")
    parser.add_argument("--successful-only", action="store_true", help="clone only demos marked success")
    return parser


def _copy_group(src: h5py.Group, dst_parent: h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            _copy_group(item, dst, key)
        else:
            dst.create_dataset(key, data=item[()], dtype=item.dtype)
    return dst


def _jitter(demo: h5py.Group, rng: np.random.Generator, noise: float, node: str | None) -> int:
    """Add Gaussian noise to the action dataset. Returns the frame count touched."""
    path = action_path(demo)
    actions = demo[path][()]
    start, stop = 0, len(actions)
    if node:
        segment = next((s for s in read_segments(demo) if s.node == node), None)
        if segment is None:
            return 0
        start, stop = segment.start, segment.end
    perturbed = actions.copy()
    perturbed[start:stop] += rng.normal(0.0, noise, size=perturbed[start:stop].shape).astype(actions.dtype)
    del demo[path]
    demo.create_dataset(path, data=perturbed)
    return stop - start


def expand(
    source: Path,
    target: Path,
    *,
    episodes: int = 10,
    noise: float = 0.01,
    seed: int = 0,
    node: str | None = None,
    include_source: bool = False,
    successful_only: bool = False,
) -> int:
    """Write ``episodes`` jittered variants of ``source`` into ``target``."""
    rng = np.random.default_rng(seed)
    target.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(source), "r") as src, h5py.File(str(target), "w") as dst:
        names = demo_names(src)
        if not names:
            raise ValueError(f"{source} has no demo_* groups")
        src_data = src[DATA_GROUP]
        if successful_only:
            names = [n for n in names if bool(src_data[n].attrs.get("success", False))]
            if not names:
                raise ValueError(f"{source} has no successful demos to clone")

        dst_data = dst.create_group(DATA_GROUP)
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value

        written = 0
        if include_source:
            for name in names:
                _copy_group(src_data[name], dst_data, f"demo_{written}")
                written += 1

        if node:
            tagged = [n for n in names if any(s.node == node for s in read_segments(src_data[n]))]
            if not tagged:
                raise ValueError(
                    f"{source}: no demo carries a segment for node {node!r}. "
                    f"Recordings made before node tagging cannot be filtered this way."
                )
            names = tagged

        for index in tqdm(range(episodes), desc="variants"):
            origin = names[index % len(names)]
            clone = _copy_group(src_data[origin], dst_data, f"demo_{written}")
            touched = _jitter(clone, rng, noise, node)
            clone.attrs["source_demo"] = origin
            clone.attrs["variant_index"] = index
            clone.attrs["mimic_noise"] = noise
            clone.attrs["mimic_node"] = node or ""
            clone.attrs["mimic_frames_jittered"] = touched
            clone.attrs.setdefault("success", True)
            # Segments survive the copy, so a downstream tool can still tell
            # which skill each frame belongs to.
            write_segments(clone, read_segments(src_data[origin]))
            written += 1

        dst_data.attrs["total"] = written
        logger.info("wrote %s demos to %s", written, target)
        return written


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[mimic] %(message)s")
    if not args.input.is_file():
        print(f"error: no recording at {args.input}")
        return 1
    written = expand(
        args.input,
        args.output,
        episodes=args.episodes,
        noise=args.noise,
        seed=args.seed,
        node=args.node,
        include_source=args.include_source,
        successful_only=args.successful_only,
    )
    print(f"wrote {written} demos to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

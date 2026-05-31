# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import argparse
import base64
import logging
import time
from pathlib import Path
from typing import Any

from annotator.config import camera_names_for_env, csv, task_description_for_env
from annotator.live import iter_live_samples
from annotator.offline import iter_hdf5_episodes, write_successful_episodes
from annotator.output import JsonlWriter
from annotator.records import FrameBundle
from annotator.vlm import VLMVerifier, _image_data_url
from common.utils import resolve_path

DEFAULT_MODEL = "Qwen/Qwen3-VL-8B-Instruct"
DEFAULT_BASE_URL = "http://localhost:8000/v1"


def _build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Annotate agentic task success with a local OpenVLLM vision model.")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="OpenAI-compatible vLLM base URL.")
    parser.add_argument("--api-key", default="EMPTY", help="API key for the OpenAI-compatible endpoint.")
    parser.add_argument("--model", default=DEFAULT_MODEL, help="Vision-language model id served by vLLM.")
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--max-tokens", type=int, default=512)
    parser.add_argument(
        "--output", default=None, help="Append JSONL annotations to this path. Always prints JSONL to stdout."
    )
    parser.add_argument("--log-level", default="INFO", choices=("DEBUG", "INFO", "WARNING", "ERROR"))
    parser.add_argument("--env", required=True, help="Environment id from config/environments/<env>.yaml.")

    subparsers = parser.add_subparsers(dest="mode", required=True)

    offline = subparsers.add_parser("offline", help="Annotate episodes from an Arena HDF5 recording.")
    offline.add_argument("--hdf5-path", required=True, help="Arena recording produced with --record-to.")
    offline.add_argument("--task-description", default=None, help="Override the environment task text.")
    offline.add_argument(
        "--cameras", default=None, help="Comma-separated obs camera keys. Defaults to RGB arrays in each demo."
    )
    offline.add_argument("--sample-frames", type=int, default=5, help="Number of frames sampled across each episode.")
    offline.add_argument("--max-episodes", type=int, default=None)
    offline.add_argument("--filter", default=None, help="Write a filtered HDF5 containing only successful episodes.")

    live = subparsers.add_parser("live", help="Annotate latest camera frames from Zenoh.")
    live.add_argument("--task-description", default=None, help="Override the environment task text.")
    live.add_argument(
        "--cameras", default=None, help="Comma-separated Zenoh camera names. Defaults to environment config."
    )
    live.add_argument("--timeout", type=float, default=30.0, help="Seconds to wait for first frames from every camera.")
    live.add_argument("--interval", type=float, default=2.0, help="Seconds between live annotations.")
    live.add_argument(
        "--count", type=int, default=1, help="Number of live snapshots to annotate. Use 0 to run forever."
    )
    live.add_argument(
        "--skip-initial-frames", type=int, default=0, help="Ignore this many live frame updates before sampling."
    )
    live.add_argument(
        "--frame-stride", type=int, default=1, help="Sample every Nth live frame update after the initial skip."
    )
    live.add_argument(
        "--dump-frames-dir",
        default=None,
        help="Live mode only: save sampled camera frames under this directory.",
    )
    live.add_argument(
        "--dump-frames-only",
        action="store_true",
        help="Live mode only: dump sampled frames and skip the VLM API call.",
    )
    live.add_argument(
        "--min-success-frames",
        type=int,
        default=0,
        help="If set, require at least this many sampled live snapshots to have annotation.success=true.",
    )
    return parser


def main() -> None:
    parser = _build_argparser()
    args = parser.parse_args()
    _resolve_path_args(args)
    logging.basicConfig(level=getattr(logging, args.log_level), format="%(message)s")
    verifier = None
    if not (args.mode == "live" and args.dump_frames_only):
        verifier = VLMVerifier(
            base_url=args.base_url,
            api_key=args.api_key,
            model=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
        )
    with JsonlWriter(args.output) as writer:
        if args.mode == "offline":
            _run_offline(args, verifier, writer)
        elif args.mode == "live":
            _run_live(args, verifier, writer)
        else:
            parser.error(f"unknown mode: {args.mode}")


def _run_offline(args: argparse.Namespace, verifier: VLMVerifier | None, writer: JsonlWriter) -> None:
    if verifier is None:
        raise ValueError("offline mode requires a VLM verifier")
    cameras = csv(args.cameras)
    task_description = task_description_for_env(args.env, args.task_description)
    records = []
    for episode in iter_hdf5_episodes(
        hdf5_path=args.hdf5_path,
        cameras=cameras,
        sample_frames=args.sample_frames,
        max_episodes=args.max_episodes,
    ):
        annotation = verifier.annotate(
            task_description=task_description,
            frames=episode.frames,
            context=f"Offline HDF5 episode {episode.episode_id} from env {args.env}.",
        )
        record = _record(
            mode="offline",
            env=args.env,
            task_description=task_description,
            annotation=annotation,
            extra={
                "episode_id": episode.episode_id,
                "total_frames": episode.total_frames,
                "sampled_frames": [frame.index for frame in episode.frames],
                "cameras": sorted(episode.frames[0].cameras) if episode.frames else [],
            },
        )
        records.append(record)
        writer.write(record)

    if args.filter:
        kept, rejected = write_successful_episodes(
            source_hdf5=args.hdf5_path,
            output_hdf5=args.filter,
            records=records,
        )
        logging.info("wrote filtered HDF5 to %s (kept %s, rejected %s)", args.filter, kept, rejected)


def _run_live(args: argparse.Namespace, verifier: VLMVerifier | None, writer: JsonlWriter) -> None:
    cameras = csv(args.cameras) or camera_names_for_env(args.env)
    task_description = task_description_for_env(args.env, args.task_description)
    count = None if args.count == 0 else args.count
    if args.min_success_frames and count is None:
        raise ValueError("--min-success-frames requires a finite --count")
    if args.dump_frames_only and args.min_success_frames:
        raise ValueError("--dump-frames-only cannot be combined with --min-success-frames")
    dump_dir = Path(args.dump_frames_dir).expanduser() if args.dump_frames_dir else None
    if args.dump_frames_only and dump_dir is None:
        raise ValueError("--dump-frames-only requires --dump-frames-dir")
    if dump_dir:
        dump_dir.mkdir(parents=True, exist_ok=True)
    successes = 0
    sampled = 0
    for frame in iter_live_samples(
        env_id=args.env,
        cameras=cameras,
        timeout=args.timeout,
        interval=args.interval,
        count=count,
        skip_initial_frames=args.skip_initial_frames,
        frame_stride=args.frame_stride,
    ):
        dumped_frames = _dump_live_frames(dump_dir, frame) if dump_dir else []
        if args.dump_frames_only:
            annotation = {
                "label": "unverified",
                "success": None,
                "confidence": 0.0,
                "reasoning": "VLM API call skipped because --dump-frames-only was set.",
                "evidence": dumped_frames,
            }
        else:
            if verifier is None:
                raise ValueError("live annotation requires a VLM verifier")
            annotation = verifier.annotate(
                task_description=task_description,
                frames=[frame],
                context=f"Live Zenoh snapshot {frame.index} from env {args.env}.",
            )
        sampled += 1
        if annotation.get("success") is True:
            successes += 1
        writer.write(
            _record(
                mode="live",
                env=args.env,
                task_description=task_description,
                annotation=annotation,
                extra={
                    "snapshot_index": frame.index,
                    "cameras": sorted(frame.cameras),
                    "sampled_snapshots": sampled,
                    "successful_snapshots": successes,
                    "dumped_frames": dumped_frames,
                },
            )
        )

    if args.min_success_frames:
        passed = successes >= args.min_success_frames
        writer.write(
            _record(
                mode="live_summary",
                env=args.env,
                task_description=task_description,
                annotation={
                    "label": "good" if passed else "bad",
                    "success": passed,
                    "confidence": 1.0,
                    "reasoning": (
                        f"{successes}/{sampled} sampled live snapshots passed; "
                        f"required at least {args.min_success_frames}."
                    ),
                    "evidence": [],
                },
                extra={
                    "sampled_snapshots": sampled,
                    "successful_snapshots": successes,
                    "min_success_frames": args.min_success_frames,
                    "skip_initial_frames": args.skip_initial_frames,
                    "frame_stride": args.frame_stride,
                    "cameras": sorted(cameras),
                },
            )
        )
        if not passed:
            raise SystemExit(1)


def _dump_live_frames(dump_dir: Path | None, frame: FrameBundle) -> list[str]:
    if dump_dir is None:
        return []
    paths: list[str] = []
    for camera_name, image in sorted(frame.cameras.items()):
        data_url = _image_data_url(image)
        encoded = data_url.partition(",")[2]
        path = dump_dir / f"snapshot_{frame.index:06d}_{camera_name}.jpg"
        path.write_bytes(base64.b64decode(encoded))
        paths.append(str(path))
    return paths


def _record(
    *,
    mode: str,
    env: str,
    task_description: str,
    annotation: dict[str, Any],
    extra: dict[str, Any],
) -> dict[str, Any]:
    return {
        "ts": time.time_ns(),
        "mode": mode,
        "env": env,
        "task_description": task_description,
        **extra,
        "annotation": annotation,
    }


def _resolve_path_args(args: argparse.Namespace) -> None:
    if args.output:
        args.output = str(resolve_path(args.output, args.env))
    if args.mode == "offline":
        args.hdf5_path = str(resolve_path(args.hdf5_path, args.env))
        if args.filter:
            args.filter = str(resolve_path(args.filter, args.env))
    elif args.mode == "live" and args.dump_frames_dir:
        args.dump_frames_dir = str(resolve_path(args.dump_frames_dir, args.env))


if __name__ == "__main__":
    main()

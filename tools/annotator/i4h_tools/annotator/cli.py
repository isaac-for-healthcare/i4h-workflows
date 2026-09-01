# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM success labelling for recorded episodes.

Offline artifact processing; no ``engine`` dependency.

``--node`` is the capability node-tagged recordings unlock: ask the model
"did the grasp succeed?" over just the grasp frames instead of "did the episode
succeed?" over all of them. A per-episode verdict cannot tell you *which* skill
failed, which is exactly what you need in order to fix anything.

Two modes: ``offline`` grades a recording, ``live`` grades whatever the running
arena is publishing on the bus. Live is for watching a rollout you have not
recorded yet — the verdict arrives while the robot is still moving.
"""

from __future__ import annotations

import argparse
import base64
import io
import logging
import os
from pathlib import Path

import h5py
import numpy as np

from i4h_common.episode import DATA_GROUP, camera_keys, demo_names, read_segments

logger = logging.getLogger("i4h_tools.annotator")

DEFAULT_MODEL = os.environ.get("I4H_ANNOTATOR_VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

PROMPT = (
    "You are grading a robot manipulation attempt. The images are evenly spaced frames "
    "from a single attempt, in order. Judge completion primarily from the visible terminal "
    "state in the final images; use earlier images as motion context. Successful placement "
    "does not require the robot to keep holding the object after releasing it at the target.\n\n"
    "Task: {task}\n\n"
    "Answer with exactly one word — SUCCESS if the task was completed, FAILURE otherwise — "
    "then a newline and one short sentence of justification."
)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-annotator", description="Label episodes with a VLM.")
    parser.add_argument("--task", required=True, help="what success looks like, in words")
    parser.add_argument("--camera", default=None, help="camera to grade; defaults to the first available")
    parser.add_argument("--frames", type=int, default=6, help="frames sampled per attempt")
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--base-url", default=os.environ.get("I4H_VLM_URL", "http://localhost:8000/v1"))
    parser.add_argument("--api-key", default=os.environ.get("OPENAI_API_KEY", "EMPTY"))
    parser.add_argument("--dry-run", action="store_true", help="sample frames and report, without calling the model")
    sub = parser.add_subparsers(dest="command", required=True)

    off = sub.add_parser("offline", help="grade a recording")
    off.add_argument("dataset", type=Path)
    off.add_argument("--node", default=None, help="grade only this workflow node's frames")
    off.add_argument("--write", action="store_true", help="write verdicts back as HDF5 attributes")
    off.add_argument("--filter", type=Path, default=None, help="write a new HDF5 of only the successful episodes")

    live = sub.add_parser("live", help="grade frames from a running arena")
    live.add_argument("--namespace", required=True, help="bus namespace, normally the workflow name")
    live.add_argument("--interval", type=float, default=2.0, help="seconds between verdicts")
    live.add_argument("--count", type=int, default=1, help="verdicts to emit; 0 runs until interrupted")
    live.add_argument("--timeout", type=float, default=30.0, help="seconds to wait for the first frame")
    return parser


def _encode(frame: np.ndarray) -> str:
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


def _sample(video: np.ndarray, count: int) -> list[np.ndarray]:
    if len(video) <= count:
        return list(video)
    indices = np.linspace(0, len(video) - 1, count).astype(int)
    return [video[i] for i in indices]


def annotate(
    dataset: Path,
    *,
    task: str,
    camera: str | None = None,
    node: str | None = None,
    frames: int = 6,
    model: str = DEFAULT_MODEL,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    write: bool = False,
    dry_run: bool = False,
) -> list[tuple[str, bool, str]]:
    """Return ``(demo_name, success, justification)`` per episode."""
    client = None
    if not dry_run:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(base_url=base_url, api_key=api_key)

    results: list[tuple[str, bool, str]] = []
    with h5py.File(str(dataset), "a" if write else "r") as handle:
        data = handle[DATA_GROUP]
        for name in demo_names(handle):
            demo = data[name]
            available = camera_keys(demo)
            if not available:
                logger.warning("%s has no camera datasets; skipping", name)
                continue
            key = camera or available[0]
            if key not in available:
                raise KeyError(f"{name}: no camera {key!r}; have {available}")

            video = demo[f"obs/{key}"][()]
            if node:
                segment = next((s for s in read_segments(demo) if s.node == node), None)
                if segment is None:
                    logger.warning("%s has no segment for node %r; skipping", name, node)
                    continue
                video = video[segment.start : segment.end]

            sampled = _sample(video, frames)
            if dry_run:
                results.append((name, False, f"dry-run: {len(sampled)} frames from {key}"))
                continue

            content: list[dict] = [{"type": "text", "text": PROMPT.format(task=task)}]
            content += [
                {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(f)}"}} for f in sampled
            ]
            reply = client.chat.completions.create(  # type: ignore[union-attr]
                model=model, messages=[{"role": "user", "content": content}], max_tokens=120, temperature=0.0
            )
            text = (reply.choices[0].message.content or "").strip()
            success = text.upper().startswith("SUCCESS")
            justification = text.split("\n", 1)[1].strip() if "\n" in text else ""
            results.append((name, success, justification))

            if write:
                attribute = f"vlm_success_{node}" if node else "vlm_success"
                demo.attrs[attribute] = success
                demo.attrs[f"{attribute}_reason"] = justification
    return results


def filter_successful(source: Path, target: Path, results: list[tuple[str, bool, str]]) -> int:
    """Write a recording holding only the episodes the model passed."""
    keep = [name for name, success, _reason in results if success]
    if not keep:
        raise ValueError("no successful episodes; refusing to write an empty recording")
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(source), "r") as src, h5py.File(str(target), "w") as dst:
        src_data = src[DATA_GROUP]
        dst_data = dst.create_group(DATA_GROUP)
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value
        for index, name in enumerate(keep):
            _copy(src_data[name], dst_data, f"demo_{index}")
        dst_data.attrs["total"] = len(keep)
        dst_data.attrs["filtered_from"] = str(source)
    logger.info("wrote %s successful demos to %s", len(keep), target)
    return len(keep)


def _copy(src: h5py.Group, dst_parent: h5py.Group, name: str) -> h5py.Group:
    dst = dst_parent.create_group(name)
    for key, value in src.attrs.items():
        dst.attrs[key] = value
    for key, item in src.items():
        if isinstance(item, h5py.Group):
            _copy(item, dst, key)
        else:
            dst.create_dataset(key, data=item[()], dtype=item.dtype)
    return dst


def annotate_live(
    *,
    namespace: str,
    task: str,
    camera: str | None = None,
    frames: int = 6,
    interval: float = 2.0,
    count: int = 1,
    timeout: float = 30.0,
    model: str = DEFAULT_MODEL,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    dry_run: bool = False,
) -> list[tuple[str, bool, str]]:
    """Grade frames off the bus while a rollout is running.

    Samples the *latest* frame ``frames`` times ``interval/frames`` apart rather
    than replaying a buffer: the point of live grading is a verdict about what
    the robot is doing now, so a stale queue would defeat it.
    """
    import time  # noqa: PLC0415

    from i4h_common.bus.base import Latest  # noqa: PLC0415
    from i4h_common.bus.keys import Keys  # noqa: PLC0415
    from i4h_common.bus.messages import CameraStream  # noqa: PLC0415
    from i4h_common.bus.zenoh_bus import open_zenoh_bus  # noqa: PLC0415

    client = None
    if not dry_run:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(base_url=base_url, api_key=api_key)

    keys = Keys(namespace)
    bus = open_zenoh_bus()
    results: list[tuple[str, bool, str]] = []
    try:
        stream = Latest(bus, keys.camera(camera) if camera else keys.camera_wildcard(), CameraStream)
        deadline = time.monotonic() + timeout
        while stream.value is None:
            if time.monotonic() > deadline:
                raise TimeoutError(f"no camera frames on {keys.root} after {timeout}s; is arena running?")
            time.sleep(0.1)

        emitted = 0
        while count == 0 or emitted < count:
            sampled: list[np.ndarray] = []
            for _ in range(frames):
                frame = stream.value
                if frame is not None:
                    sampled.append(np.frombuffer(frame.data, dtype=np.uint8).reshape(frame.height, frame.width, -1))
                time.sleep(max(interval / max(frames, 1), 0.0))

            label = f"live_{emitted}"
            if dry_run or not sampled:
                results.append((label, False, f"dry-run: {len(sampled)} frames"))
            else:
                results.append((label, *_grade(client, model, task, sampled)))
            print(f"{results[-1][0]:<12} {'SUCCESS' if results[-1][1] else 'FAILURE'}  {results[-1][2]}")
            emitted += 1
    except KeyboardInterrupt:
        logger.info("stopped after %s verdicts", len(results))
    finally:
        bus.close()
    return results


def _grade(client, model: str, task: str, sampled: list[np.ndarray]) -> tuple[bool, str]:
    """One VLM call over the sampled frames."""
    content: list[dict] = [{"type": "text", "text": PROMPT.format(task=task)}]
    content += [{"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(f)}"}} for f in sampled]
    reply = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], max_tokens=120, temperature=0.0
    )
    text = (reply.choices[0].message.content or "").strip()
    return text.upper().startswith("SUCCESS"), text.split("\n", 1)[1].strip() if "\n" in text else ""


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[annotator] %(message)s")

    if args.command == "live":
        annotate_live(
            namespace=args.namespace,
            task=args.task,
            camera=args.camera,
            frames=args.frames,
            interval=args.interval,
            count=args.count,
            timeout=args.timeout,
            model=args.model,
            base_url=args.base_url,
            api_key=args.api_key,
            dry_run=args.dry_run,
        )
        return 0

    if not args.dataset.is_file():
        print(f"error: no recording at {args.dataset}")
        return 1
    results = annotate(
        args.dataset,
        task=args.task,
        camera=args.camera,
        node=args.node,
        frames=args.frames,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        write=args.write,
        dry_run=args.dry_run,
    )
    for name, success, reason in results:
        print(f"{name:<12} {'SUCCESS' if success else 'FAILURE'}  {reason}")
    passed = sum(1 for _n, s, _r in results if s)
    print(f"\n{passed}/{len(results)} succeeded")
    if args.filter is not None:
        kept = filter_successful(args.dataset, args.filter, results)
        print(f"wrote {kept} successful demos to {args.filter}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

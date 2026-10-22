# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""VLM success labelling for recorded episodes.

Offline artifact processing; no ``engine`` dependency.

``--node`` is the capability node-tagged recordings unlock: ask the model
"did the grasp succeed?" over just the grasp frames instead of "did the episode
succeed?" over all of them. A per-episode verdict cannot tell you *which* skill
failed, which is exactly what you need in order to fix anything.

What to show the model is a property of the task, not of this tool. A
placement is judged by where the object ends up (``--judge terminal``, the
default); a scan or sweep is judged by whether the motion happens at all
(``--judge trajectory``), because its end state looks much like its start.
``--camera`` may name several cameras when no single view shows the task. A
scene declares these choices in its manifest's ``annotate`` block, and
``scripts/e2e/run.sh`` passes them here.

Episodes whose sampled frames are identical get one verdict: ``tools/mimic``
copies a source demo's camera frames into each variant and jitters only the
actions, so a second call would grade the same pictures again. The summary
counts verdicts graded by the model apart from those reused this way.
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import io
import logging
import os
from collections.abc import Sequence
from pathlib import Path
from typing import NamedTuple

import h5py
import numpy as np

from i4h_common.episode import DATA_GROUP, RECORDING_VERSION, camera_keys, demo_names, read_segments, recording_version

logger = logging.getLogger("i4h_tools.annotator")

DEFAULT_MODEL = os.environ.get("I4H_ANNOTATOR_VLLM_MODEL", "Qwen/Qwen3-VL-8B-Instruct")

_INTRO = (
    "You are grading a robot manipulation attempt. The images are evenly spaced frames from a single attempt, in order."
)
_CAMERAS = " They come from {count} cameras: each camera's frames follow its label and cover the same instants."
#: How to read the frames, by ``--judge``.
JUDGES = {
    # A placement: the object's final resting place is the evidence.
    "terminal": (
        " Judge completion primarily from the visible terminal state in the final images; use earlier images as"
        " motion context. Successful placement does not require the robot to keep holding the object after"
        " releasing it at the target."
    ),
    # A scan or sweep: the motion is the evidence, and the robot may lift away
    # or hold still once it is done, so the last frame proves little.
    "trajectory": (
        " The task is an action to perform during the attempt rather than a state to leave behind, so judge"
        " whether that action visibly happens across the images. The robot may hold still or withdraw near the"
        " end; that does not undo an action it already completed."
    ),
}
_QUESTION = (
    "\n\nTask: {task}\n\n"
    "Answer with exactly one word — SUCCESS if the task was completed, FAILURE otherwise — "
    "then a newline and one short sentence of justification."
)


def build_prompt(task: str, *, judge: str = "terminal", cameras: int = 1) -> str:
    """The grading instruction for ``task`` over frames from ``cameras`` cameras."""
    if judge not in JUDGES:
        raise ValueError(f"unknown judge {judge!r}; expected one of {sorted(JUDGES)}")
    camera_note = _CAMERAS.format(count=cameras) if cameras > 1 else ""
    return _INTRO + camera_note + JUDGES[judge] + _QUESTION.format(task=task)


#: The single-camera placement prompt, unchanged since the annotator shipped.
PROMPT = build_prompt("{task}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-annotator", description="Label episodes with a VLM.")
    parser.add_argument("--task", required=True, help="what success looks like, in words")
    parser.add_argument(
        "--camera",
        action="append",
        default=None,
        help="camera to grade; repeat or comma-separate to show several (default: the first available)",
    )
    parser.add_argument(
        "--judge",
        choices=sorted(JUDGES),
        default="terminal",
        help="terminal: judge the end state (placements); trajectory: judge whether the action happens (scans)",
    )
    parser.add_argument("--frames", type=int, default=6, help="frames sampled per attempt, per camera")
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
    return parser


def _encode(frame: np.ndarray) -> str:
    from PIL import Image  # noqa: PLC0415

    buffer = io.BytesIO()
    Image.fromarray(frame.astype(np.uint8)).save(buffer, format="JPEG", quality=85)
    return base64.b64encode(buffer.getvalue()).decode()


class Verdict(NamedTuple):
    """One episode's grade."""

    name: str
    success: bool
    reason: str
    #: The episode whose identical frames earned this verdict; empty when the model graded this one.
    reused_from: str = ""


def _sample_indices(length: int, count: int) -> list[int]:
    """``count`` evenly spaced frame indices into ``length`` frames, ascending and distinct."""
    if length <= count:
        return list(range(length))
    return np.linspace(0, length - 1, count).astype(int).tolist()


def _read_frames(video: h5py.Dataset, count: int, start: int = 0, stop: int | None = None) -> list[np.ndarray]:
    """The sampled frames of ``video[start:stop]``, read without loading the rest of it."""
    start, stop, _ = slice(start, stop).indices(len(video))
    indices = [start + index for index in _sample_indices(stop - start, count)]
    return list(video[indices]) if indices else []


def camera_list(value: str | Sequence[str] | None) -> list[str]:
    """``--camera`` values as names: repeated flags and comma-separated lists both work."""
    if value is None:
        return []
    items = [value] if isinstance(value, str) else list(value)
    return [name.strip() for item in items for name in item.split(",") if name.strip()]


def build_content(task: str, frames: dict[str, list[np.ndarray]], *, judge: str = "terminal") -> list[dict]:
    """One chat message: the instruction, then each camera's frames behind its label."""
    content: list[dict] = [{"type": "text", "text": build_prompt(task, judge=judge, cameras=len(frames))}]
    for name, sampled in frames.items():
        if len(frames) > 1:
            content.append({"type": "text", "text": f"Camera {name}:"})
        content += [
            {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{_encode(f)}"}} for f in sampled
        ]
    return content


def _fingerprint(frames: dict[str, list[np.ndarray]]) -> str:
    digest = hashlib.sha256()
    for name, sampled in frames.items():
        digest.update(name.encode())
        for frame in sampled:
            digest.update(np.ascontiguousarray(frame).tobytes())
    return digest.hexdigest()


def _ask(client, model: str, content: list[dict]) -> tuple[bool, str]:
    """One VLM call; returns ``(success, justification)``."""
    reply = client.chat.completions.create(
        model=model, messages=[{"role": "user", "content": content}], max_tokens=120, temperature=0.0
    )
    text = (reply.choices[0].message.content or "").strip()
    return text.upper().startswith("SUCCESS"), text.split("\n", 1)[1].strip() if "\n" in text else ""


def annotate(
    dataset: Path,
    *,
    task: str,
    camera: str | Sequence[str] | None = None,
    node: str | None = None,
    frames: int = 6,
    judge: str = "terminal",
    model: str = DEFAULT_MODEL,
    base_url: str = "http://localhost:8000/v1",
    api_key: str = "EMPTY",
    write: bool = False,
    dry_run: bool = False,
) -> list[Verdict]:
    """Return one :class:`Verdict` per episode."""
    if judge not in JUDGES:
        raise ValueError(f"unknown judge {judge!r}; expected one of {sorted(JUDGES)}")
    if frames < 1:
        raise ValueError(f"frames must be at least 1, got {frames}")
    requested = camera_list(camera)
    client = None
    if not dry_run:
        from openai import OpenAI  # noqa: PLC0415

        client = OpenAI(base_url=base_url, api_key=api_key)

    results: list[Verdict] = []
    #: Verdict per distinct set of sampled frames, and the demo that earned it.
    graded: dict[str, Verdict] = {}
    unversioned: list[str] = []
    with h5py.File(str(dataset), "a" if write else "r") as handle:
        data = handle[DATA_GROUP]
        for name in demo_names(handle):
            demo = data[name]
            if recording_version(demo) < RECORDING_VERSION:
                unversioned.append(name)
            available = camera_keys(demo)
            if not available:
                logger.warning("%s has no camera datasets; skipping", name)
                continue
            keys = requested or available[:1]
            missing = [key for key in keys if key not in available]
            if missing:
                raise KeyError(f"{name}: no camera {missing[0]!r}; have {available}")

            segment = None
            if node:
                segment = next((s for s in read_segments(demo) if s.node == node), None)
                if segment is None:
                    logger.warning("%s has no segment for node %r; skipping", name, node)
                    continue
            sampled: dict[str, list[np.ndarray]] = {}
            for key in keys:
                # Only the sampled frames leave the file: a long recording's
                # full video, times every camera, need not fit in memory.
                bounds = (segment.start, segment.end) if segment is not None else (0, None)
                sampled[key] = _read_frames(demo[f"obs/{key}"], frames, *bounds)

            if dry_run:
                counts = ", ".join(f"{len(v)} frames from {k}" for k, v in sampled.items())
                results.append(Verdict(name, False, f"dry-run: {counts} ({judge})"))
                continue

            fingerprint = _fingerprint(sampled)
            source = graded.get(fingerprint)
            if source is None:
                success, justification = _ask(client, model, build_content(task, sampled, judge=judge))
                verdict = graded[fingerprint] = Verdict(name, success, justification)
            else:
                # A mimic variant: its frames are its source's, so the verdict is too.
                logger.info("%s shows the same frames as %s; reusing its verdict", name, source.name)
                verdict = Verdict(name, source.success, source.reason, reused_from=source.name)
            results.append(verdict)

            if write:
                attribute = f"vlm_success_{node}" if node else "vlm_success"
                demo.attrs[attribute] = verdict.success
                demo.attrs[f"{attribute}_reason"] = verdict.reason
                if verdict.reused_from:
                    demo.attrs[f"{attribute}_from"] = verdict.reused_from
                elif f"{attribute}_from" in demo.attrs:
                    del demo.attrs[f"{attribute}_from"]
    if unversioned:
        logger.warning(
            "%d demo(s) predate recording_version %d (%s): one whose episode ended through a termination "
            "term ends on a frame of the next episode; re-record before trusting its final frames",
            len(unversioned),
            RECORDING_VERSION,
            ", ".join(unversioned[:5]) + (", ..." if len(unversioned) > 5 else ""),
        )
    return results


def filter_successful(source: Path, target: Path, results: Sequence[Sequence]) -> int:
    """Write a recording holding only the episodes the model passed.

    Kept demos are renumbered from ``demo_0``. Each records its name in the
    source as ``filtered_from_demo``, and a ``vlm_success*_from`` reference is
    rewritten to the new name of the demo it names.
    """
    keep = [result[0] for result in results if result[1]]
    if not keep:
        raise ValueError("no successful episodes; refusing to write an empty recording")
    renamed = {name: f"demo_{index}" for index, name in enumerate(keep)}
    target.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(str(source), "r") as src, h5py.File(str(target), "w") as dst:
        src_data = src[DATA_GROUP]
        # File-root attributes too, such as Isaac Lab's format_version.
        for key, value in src.attrs.items():
            dst.attrs[key] = value
        dst_data = dst.create_group(DATA_GROUP)
        for key, value in src_data.attrs.items():
            dst_data.attrs[key] = value
        for name, new_name in renamed.items():
            demo = _copy(src_data[name], dst_data, new_name)
            demo.attrs["filtered_from_demo"] = name
            for key in [key for key in demo.attrs if key.startswith("vlm_success") and key.endswith("_from")]:
                referenced = demo.attrs[key]
                referenced = referenced.decode() if isinstance(referenced, bytes) else str(referenced)
                if referenced in renamed:
                    demo.attrs[key] = renamed[referenced]
                else:
                    # Its verdict came from a demo this file does not hold; the
                    # old name would point at whichever demo now carries it.
                    del demo.attrs[key]
                    logger.warning("%s: removed %s reference to %s, absent from filtered output", name, key, referenced)
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


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[annotator] %(message)s")

    cameras = camera_list(args.camera)
    if not args.dataset.is_file():
        print(f"error: no recording at {args.dataset}")
        return 1
    results = annotate(
        args.dataset,
        task=args.task,
        camera=cameras or None,
        node=args.node,
        frames=args.frames,
        judge=args.judge,
        model=args.model,
        base_url=args.base_url,
        api_key=args.api_key,
        write=args.write,
        dry_run=args.dry_run,
    )
    for verdict in results:
        print(f"{verdict.name:<12} {'SUCCESS' if verdict.success else 'FAILURE'}  {verdict.reason}")
    passed = sum(1 for verdict in results if verdict.success)
    split = ""
    if not args.dry_run:
        # Reused verdicts are mimic variants no model looked at; say how many.
        reused = sum(1 for verdict in results if verdict.reused_from)
        split = f" ({len(results) - reused} graded, {reused} reused)"
    print(f"\n{passed}/{len(results)} succeeded{split}")
    if args.filter is not None:
        kept = filter_successful(args.dataset, args.filter, results)
        print(f"wrote {kept} successful demos to {args.filter}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

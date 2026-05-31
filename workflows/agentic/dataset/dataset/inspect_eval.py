# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Diagnostic CLI for an Agentic eval recording.

Decodes the action layout written by ``arena/run.sh ... --record-to`` and
reports the slot-by-slot stats (joint targets, locomotion commands), forward
translation, and per-joint state range. Use when a finetuned policy fails
to drive the robot and you want to see whether the policy actually emitted
locomotion commands or if the recorded actions are degenerate.

Usage::

  uv --directory workflows/agentic/dataset run i4h-agentic-eval-inspect \\
      --env <env_id> \\
      runs/eval_<env_id>_*/data/verify.hdf5
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

import h5py
import numpy as np

logger = logging.getLogger("dataset.inspect_eval")


# Known action-tensor layouts: (slot_name, start, end).
_ACTION_LAYOUTS: dict[str, list[tuple[str, int, int]]] = {
    # G1 g1_wbc_joint embodiment (50D):
    #   [0:43] joint targets, [43:46] navigate_cmd, [46] base_height, [47:50] torso_rpy
    "g1_wbc_joint_50d": [
        ("joints", 0, 43),
        ("navigate_command", 43, 46),
        ("base_height_command", 46, 47),
        ("torso_orientation_rpy_command", 47, 50),
    ],
    # SO-ARM 101 joint position (6D arm + 1D gripper).
    "so_arm_7d": [
        ("arm_joints", 0, 6),
        ("gripper", 6, 7),
    ],
}

# Key on recorded action_dim so new envs work when their action shape is known.
_DIM_TO_LAYOUT: dict[int, str] = {
    50: "g1_wbc_joint_50d",  # All G1 locomanip envs (current and future)
    7: "so_arm_7d",
}


def _detect_layout(env: str | None, action_dim: int) -> list[tuple[str, int, int]] | None:
    # 1. Dim-based heuristic — works for any env that shares the embodiment.
    layout_key = _DIM_TO_LAYOUT.get(action_dim)
    if layout_key is not None:
        return _ACTION_LAYOUTS[layout_key]
    # 2. If we still can't classify, return None so the caller dumps the
    #    raw per-dim summary instead of guessing. Caller passes ``--env <id>``
    #    only to surface it in the header.
    _ = env
    return None


def _summarize_slot(arr: np.ndarray, name: str, start: int, end: int) -> str:
    slot = arr[:, start:end]
    if slot.size == 0:
        return f"  {name:32s} [{start}:{end}] empty"
    rng = slot.max(axis=0) - slot.min(axis=0)
    if slot.shape[1] == 1:
        return (
            f"  {name:32s} [{start}:{end}] "
            f"min={slot.min():.4f} max={slot.max():.4f} mean={slot.mean():.4f} std={slot.std():.4f}"
        )
    return (
        f"  {name:32s} [{start}:{end}] "
        f"max={slot.max(axis=0).round(4).tolist()} "
        f"mean={slot.mean(axis=0).round(4).tolist()} "
        f"max_range={rng.max():.4f}"
    )


def _forward_translation(d: h5py.Group) -> tuple[float, float, float] | None:
    if "obs/robot_pos" not in d:
        return None
    pos = d["obs/robot_pos"][:]
    if pos.ndim < 2 or pos.shape[0] < 2:
        return None
    dx = float(pos[-1, 0] - pos[0, 0])
    dy = float(pos[-1, 1] - pos[0, 1])
    return dx, dy, float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))


def _joint_motion(d: h5py.Group, threshold: float = 0.05) -> list[tuple[int, float, float, float]]:
    """Return list of (index, range, start, end) for joints whose range >= threshold."""
    if "obs/robot_joint_pos" not in d:
        return []
    j = d["obs/robot_joint_pos"][:]
    if j.ndim < 2:
        return []
    out: list[tuple[int, float, float, float]] = []
    rng = j.max(axis=0) - j.min(axis=0)
    for i, r in enumerate(rng):
        if float(r) >= threshold:
            out.append((i, float(r), float(j[0, i]), float(j[-1, i])))
    return out


def inspect_eval(path: Path, env: str | None) -> None:
    print(f"=== {path} ===")
    with h5py.File(path, "r") as f:
        if "data" not in f:
            print("  no 'data/' group; not an Agentic eval recording")
            return
        demos = sorted(f["data"].keys())
        print(f"demos: {len(demos)}  env={env or '(unspecified)'}")
        for demo in demos:
            d = f[f"data/{demo}"]
            if "actions" not in d:
                print(f"\n--- {demo} ---")
                print("  no 'actions' dataset")
                continue
            a = d["actions"][:]
            print(f"\n--- {demo} ---")
            print(f"  steps: {a.shape[0]}    action dim: {a.shape[1]}")

            layout = _detect_layout(env, a.shape[1])
            if layout is None:
                print(f"  no known layout for action_dim={a.shape[1]}; pass --env to enable decode")
                print(f"  raw action max_range per dim: " f"{(a.max(axis=0)-a.min(axis=0)).max():.4f}")
            else:
                print("  action layout:")
                for name, start, end in layout:
                    if end > a.shape[1]:
                        continue
                    print(_summarize_slot(a, name, start, end))

            fwd = _forward_translation(d)
            if fwd is not None:
                dx, dy, dist = fwd
                walked = "YES" if dx > 0.3 else ("partial" if dx > 0.1 else "no")
                print(f"  world translation: dx={dx:+.3f} m  dy={dy:+.3f} m  total={dist:.3f} m  walked? {walked}")
            else:
                print("  world translation: obs/robot_pos missing (cannot evaluate)")

            motion = _joint_motion(d)
            if motion:
                top = sorted(motion, key=lambda t: -t[1])[:8]
                print(f"  joint state ranges (top {len(top)} by range, threshold 0.05):")
                for i, r, s, e in top:
                    print(f"    dim {i:>2}  range={r:.3f}  start={s:+.3f}  end={e:+.3f}")
            else:
                print("  no joints moved by >=0.05 rad")


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Inspect an Agentic eval verify.hdf5 recording.")
    parser.add_argument("path", help="Path to verify.hdf5 (or any Agentic eval HDF5 file).")
    parser.add_argument(
        "--env",
        default=None,
        help="Env id used to look up the action-tensor layout. Defaults to dim-based heuristic if omitted.",
    )
    return parser


def main() -> None:
    logging.basicConfig(level=logging.WARNING, format="[%(levelname)s] %(message)s")
    args = build_argparser().parse_args()
    path = Path(args.path).expanduser().resolve()
    if not path.exists():
        raise FileNotFoundError(path)
    inspect_eval(path, args.env)


if __name__ == "__main__":
    main()

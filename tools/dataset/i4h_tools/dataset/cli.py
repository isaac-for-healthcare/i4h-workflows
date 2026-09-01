# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""HDF5 → LeRobot conversion, recording inspection, and action diagnostics."""

from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import h5py
import numpy as np
from tqdm import tqdm

from i4h_common.config import get_robot_config
from i4h_common.episode import DATA_GROUP, Episode, camera_keys, demo_names
from i4h_common.joint_utils import isaaclab_rad_to_lerobot

logger = logging.getLogger("i4h_tools.dataset")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-dataset", description="Convert and inspect workflow recordings.")
    sub = parser.add_subparsers(dest="command", required=True)

    convert_cmd = sub.add_parser("convert", help="HDF5 → LeRobot")
    convert_cmd.add_argument("input", type=Path)
    convert_cmd.add_argument("output", type=Path)
    convert_cmd.add_argument("--robot", required=True, help="embodiment descriptor name")
    convert_cmd.add_argument("--repo-id", default=None, help="LeRobot repo id; defaults to the output dir name")
    convert_cmd.add_argument("--fps", type=int, default=30)
    convert_cmd.add_argument(
        "--video-codec",
        choices=("h264", "hevc", "libsvtav1"),
        default="h264",
        help="video codec; h264 is the default because GR00T's decord backend cannot read AV1 reliably",
    )
    convert_cmd.add_argument("--skip-frames", type=int, default=0, help="drop N leading frames per demo")
    convert_cmd.add_argument("--successful-only", action="store_true")
    convert_cmd.add_argument("--task", default="", help="natural-language task string stored per frame")
    convert_cmd.add_argument(
        "--g1-wbc-policy-actions",
        action="store_true",
        help="map a 23D G1 Pink teleop command plus 43D measured state to the 50D G1 policy action contract",
    )

    inspect_cmd = sub.add_parser("inspect", help="summarise a recording")
    inspect_cmd.add_argument("input", type=Path)
    inspect_cmd.add_argument("--segments", action="store_true", help="show per-node frame ranges")

    actions_cmd = sub.add_parser("actions", help="decode the action tensor slot by slot")
    actions_cmd.add_argument("input", type=Path)
    actions_cmd.add_argument("--threshold", type=float, default=0.05, help="joint range to count as motion")
    return parser


def inspect(path: Path, *, show_segments: bool = False) -> None:
    with h5py.File(str(path), "r") as handle:
        names = demo_names(handle)
        data = handle[DATA_GROUP]
        print(f"{path}: {len(names)} demos")
        for name in names:
            episode = Episode(name, data[name])
            cameras = ", ".join(episode.cameras) or "none"
            flag = "ok " if episode.success else "FAIL"
            summary = (
                f"  {name:<12} {flag} {episode.num_samples:>5} frames  "
                f"actions{episode.actions.shape}  cameras: {cameras}"
            )
            print(summary)
            if show_segments:
                for segment in episode.segments:
                    print(f"      {segment.node:<20} {segment.task_id:<34} [{segment.start}:{segment.end}]")


#: Action-tensor layouts, keyed by recorded action dim so any env sharing the
#: embodiment decodes without being named.
ACTION_LAYOUTS: dict[int, list[tuple[str, int, int]]] = {
    # G1 whole-body: [0:43] joints, [43:46] navigate, [46] base height, [47:50] torso rpy.
    50: [
        ("joints", 0, 43),
        ("navigate_command", 43, 46),
        ("base_height_command", 46, 47),
        ("torso_orientation_rpy_command", 47, 50),
    ],
    # SO-ARM 101: 6 arm joints + jaw.
    7: [("arm_joints", 0, 6), ("gripper", 6, 7)],
}

G1_WBC_ACTION_NAMES = (
    "navigate_command.x",
    "navigate_command.y",
    "navigate_command.yaw",
    "base_height_command",
    "torso_orientation_rpy_command.roll",
    "torso_orientation_rpy_command.pitch",
    "torso_orientation_rpy_command.yaw",
)


def _column_names(config, kind: str, width: int) -> list[str]:
    """Return truthful column labels for the tensor actually recorded."""
    declared = list(getattr(config, f"{kind}_names"))
    if len(declared) == width:
        return declared
    if kind == "action" and len(declared) + len(G1_WBC_ACTION_NAMES) == width:
        return declared + list(G1_WBC_ACTION_NAMES)
    logger.warning(
        "%s descriptor has %s %s names for a %s-D tensor; using positional names",
        config.name,
        len(declared),
        kind,
        width,
    )
    return [f"{kind}_{index}" for index in range(width)]


def _g1_wbc_policy_actions(actions: np.ndarray, states: np.ndarray | None) -> np.ndarray:
    """Lift G1 Pink teleop commands into the joint-plus-WBC policy contract."""
    if actions.ndim != 2 or actions.shape[1] != 23:
        raise ValueError(f"G1 WBC policy-action mapping requires actions(N, 23), got {actions.shape}")
    if states is None or states.ndim != 2 or states.shape != (actions.shape[0], 43):
        shape = None if states is None else states.shape
        raise ValueError(f"G1 WBC policy-action mapping requires states(N, 43), got {shape}")
    return np.concatenate(
        (
            states,
            actions[:, 16:19],
            actions[:, 19:20],
            actions[:, 20:23],
        ),
        axis=1,
        dtype=np.float32,
    )


def _to_policy_joint_coordinates(values: np.ndarray, config) -> np.ndarray:
    """Map a pure joint-position tensor to the policy's calibrated coordinates."""
    isaaclab_range = config.isaaclab_joint_pos_limit_range
    lerobot_range = config.lerobot_joint_pos_limit_range
    if not isaaclab_range and not lerobot_range:
        return values
    if not isaaclab_range or not lerobot_range:
        raise ValueError(f"{config.name}: both IsaacLab and LeRobot joint ranges are required")
    if values.shape[-1] != len(isaaclab_range):
        return values
    return isaaclab_rad_to_lerobot(values, isaaclab_range, lerobot_range).astype(np.float32)


def _write_g1_wbc_modality(target: Path) -> Path:
    """Write the GR00T-specific semantic slices for a converted G1 dataset."""
    modality = {
        "state": {
            "waist": {"start": 12, "end": 15},
            "left_arm": {"start": 15, "end": 22},
            "right_arm": {"start": 22, "end": 29},
            "left_hand": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
        },
        "action": {
            "left_arm": {"start": 15, "end": 22},
            "right_arm": {"start": 22, "end": 29},
            "left_hand": {"start": 29, "end": 36},
            "right_hand": {"start": 36, "end": 43},
            "navigate_command": {"start": 43, "end": 46},
            "base_height_command": {"start": 46, "end": 47},
        },
        "video": {
            "ego_view": {"original_key": "observation.images.head"},
            "room": {"original_key": "observation.images.room"},
        },
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    path = target / "meta" / "modality.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(modality, handle, indent=2)
        handle.write("\n")
    return path


def _write_split_modality(target: Path, *, config, cameras: tuple[str, ...]) -> Path:
    """Write GR00T semantic slices declared by an embodiment descriptor."""
    modality = {
        "state": {name: {"start": start, "end": end} for name, start, end in config.state_split},
        "action": {name: {"start": start, "end": end} for name, start, end in config.action_split},
        "video": {camera: {"original_key": f"observation.images.{camera}"} for camera in cameras},
        "annotation": {
            "human.task_description": {"original_key": "task_index"},
        },
    }
    path = target / "meta" / "modality.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(modality, handle, indent=2)
        handle.write("\n")
    return path


def _uses_declared_modality(*, config, state_width: int, action_width: int) -> bool:
    """Whether descriptor splits completely cover the converted tensors."""

    def covers(splits: tuple[tuple[str, int, int], ...], width: int) -> bool:
        cursor = 0
        for _name, start, end in splits:
            if start != cursor or end <= start:
                return False
            cursor = end
        return bool(splits) and cursor == width

    return covers(config.state_split, state_width) and covers(config.action_split, action_width)


def _uses_g1_wbc_modality(*, robot: str, state_width: int, action_width: int) -> bool:
    """Whether the converted tensors already match the native G1 WBC policy contract."""
    return robot == "g1" and state_width == 43 and action_width == 50


def _write_dataset_stats(target: Path, stats: dict) -> Path:
    """Materialize aggregate statistics required by GR00T's LeRobot loader."""
    from lerobot.common.datasets.utils import write_stats

    write_stats(stats, target)
    return target / "meta" / "stats.json"


def _slot(values: np.ndarray, name: str, start: int, end: int) -> str:
    slot = values[:, start:end]
    if slot.size == 0:
        return f"  {name:32s} [{start}:{end}] empty"
    if slot.shape[1] == 1:
        return (
            f"  {name:32s} [{start}:{end}] min={slot.min():.4f} max={slot.max():.4f} "
            f"mean={slot.mean():.4f} std={slot.std():.4f}"
        )
    spread = slot.max(axis=0) - slot.min(axis=0)
    return (
        f"  {name:32s} [{start}:{end}] max={slot.max(axis=0).round(4).tolist()} "
        f"mean={slot.mean(axis=0).round(4).tolist()} max_range={spread.max():.4f}"
    )


def _translation(demo: h5py.Group) -> tuple[float, float, float] | None:
    if "obs/robot_pos" not in demo:
        return None
    pos = demo["obs/robot_pos"][()]
    if pos.ndim < 2 or pos.shape[0] < 2:
        return None
    return float(pos[-1, 0] - pos[0, 0]), float(pos[-1, 1] - pos[0, 1]), float(np.linalg.norm(pos[-1, :2] - pos[0, :2]))


def _moving_joints(demo: h5py.Group, threshold: float) -> list[tuple[int, float, float, float]]:
    if "obs/robot_joint_pos" not in demo:
        return []
    joints = demo["obs/robot_joint_pos"][()]
    if joints.ndim < 2:
        return []
    spread = joints.max(axis=0) - joints.min(axis=0)
    return [
        (i, float(r), float(joints[0, i]), float(joints[-1, i])) for i, r in enumerate(spread) if float(r) >= threshold
    ]


def actions(path: Path, *, threshold: float = 0.05) -> None:
    """Decode the recorded action tensor.

    For when a policy runs but the robot does not move: it separates "the policy
    emitted nothing" from "the policy emitted commands the embodiment ignored",
    which the frame counts in ``inspect`` cannot distinguish.
    """
    with h5py.File(str(path), "r") as handle:
        if DATA_GROUP not in handle:
            print(f"{path}: no {DATA_GROUP}/ group; not a recording")
            return
        names = demo_names(handle)
        print(f"{path}: {len(names)} demos")
        for name in names:
            demo = handle[DATA_GROUP][name]
            if "actions" not in demo:
                print(f"\n--- {name} ---\n  no actions dataset")
                continue
            values = demo["actions"][()]
            print(f"\n--- {name} ---")
            print(f"  steps: {values.shape[0]}    action dim: {values.shape[1]}")

            layout = ACTION_LAYOUTS.get(values.shape[1])
            if layout is None:
                spread = (values.max(axis=0) - values.min(axis=0)).max()
                print(f"  no known layout for action_dim={values.shape[1]}; raw max_range={spread:.4f}")
            else:
                print("  action layout:")
                for slot_name, start, end in layout:
                    if end <= values.shape[1]:
                        print(_slot(values, slot_name, start, end))

            moved = _translation(demo)
            if moved is None:
                print("  world translation: obs/robot_pos missing")
            else:
                dx, dy, total = moved
                walked = "YES" if total > 0.3 else ("partial" if total > 0.1 else "no")
                print(f"  world translation: dx={dx:+.3f} m dy={dy:+.3f} m total={total:.3f} m  walked? {walked}")

            motion = sorted(_moving_joints(demo, threshold), key=lambda t: -t[1])[:8]
            if motion:
                print(f"  joint ranges above {threshold} (top {len(motion)}):")
                for index, spread, first, last in motion:
                    print(f"    joint {index:<3} range={spread:.4f} start={first:+.4f} end={last:+.4f}")
            else:
                print(f"  no joint moved more than {threshold}")


def convert(
    source: Path,
    target: Path,
    *,
    robot: str,
    repo_id: str | None = None,
    fps: int = 30,
    video_codec: str = "h264",
    skip_frames: int = 0,
    successful_only: bool = False,
    task: str = "",
    g1_wbc_policy_actions: bool = False,
) -> int:
    """Write a LeRobot dataset. Returns the number of episodes written."""
    from lerobot.common.datasets.lerobot_dataset import LeRobotDataset
    from lerobot.common.datasets.video_utils import encode_video_frames

    class _CodecLeRobotDataset(LeRobotDataset):
        """Use the caller-selected codec instead of LeRobot's hard-coded AV1 default."""

        def encode_episode_videos(self, episode_index: int) -> dict:
            video_paths = {}
            for key in self.meta.video_keys:
                video_path = self.root / self.meta.get_video_file_path(episode_index, key)
                video_paths[key] = str(video_path)
                if video_path.is_file():
                    continue
                image_dir = self._get_image_file_path(
                    episode_index=episode_index,
                    image_key=key,
                    frame_index=0,
                ).parent
                encode_video_frames(
                    image_dir,
                    video_path,
                    self.fps,
                    vcodec=video_codec,
                    overwrite=True,
                )
            return video_paths

    config = get_robot_config(robot)
    target.parent.mkdir(parents=True, exist_ok=True)

    with h5py.File(str(source), "r") as handle:
        names = demo_names(handle)
        data = handle[DATA_GROUP]
        if successful_only:
            names = [n for n in names if bool(data[n].attrs.get("success", False))]
        if not names:
            raise ValueError(f"{source}: nothing to convert")

        sample_episode = Episode(names[0], data[names[0]])
        sample_states = sample_episode.states
        sample_actions = sample_episode.actions
        if g1_wbc_policy_actions:
            if robot != "g1":
                raise ValueError("--g1-wbc-policy-actions requires --robot g1")
            sample_actions = _g1_wbc_policy_actions(sample_actions, sample_states)
        sample_states = _to_policy_joint_coordinates(sample_states, config) if sample_states is not None else None
        sample_actions = _to_policy_joint_coordinates(sample_actions, config)
        action_width = int(sample_actions.shape[-1])
        state_width = int(sample_states.shape[-1]) if sample_states is not None else action_width
        cameras = camera_keys(data[names[0]])
        features = {
            "observation.state": {
                "dtype": "float32",
                "shape": (state_width,),
                "names": _column_names(config, "state", state_width),
            },
            "action": {
                "dtype": "float32",
                "shape": (action_width,),
                "names": _column_names(config, "action", action_width),
            },
        }
        for camera in cameras:
            sample = data[names[0]][f"obs/{camera}"]
            features[f"observation.images.{camera}"] = {
                "dtype": "video",
                "shape": tuple(int(v) for v in sample.shape[1:]),
                "names": ["height", "width", "channel"],
            }

        dataset = _CodecLeRobotDataset.create(
            repo_id=repo_id or target.name,
            fps=fps,
            root=target,
            features=features,
            use_videos=bool(cameras),
        )

        for name in tqdm(names, desc="episodes"):
            episode = Episode(name, data[name])
            states = episode.states
            actions = episode.actions
            if g1_wbc_policy_actions:
                actions = _g1_wbc_policy_actions(actions, states)
            states = _to_policy_joint_coordinates(states, config) if states is not None else None
            actions = _to_policy_joint_coordinates(actions, config)
            actions = actions[skip_frames:]
            states = states[skip_frames:] if states is not None else actions
            if actions.shape[-1] != action_width:
                raise ValueError(
                    f"{name}: action width {actions.shape[-1]} differs from first episode's {action_width}"
                )
            if states.shape[-1] != state_width:
                raise ValueError(f"{name}: state width {states.shape[-1]} differs from first episode's {state_width}")
            videos = {c: data[name][f"obs/{c}"][()][skip_frames:] for c in cameras}
            for index in range(len(actions)):
                frame = {
                    "observation.state": np.asarray(states[index], dtype=np.float32),
                    "action": np.asarray(actions[index], dtype=np.float32),
                }
                for camera, video in videos.items():
                    frame[f"observation.images.{camera}"] = video[index]
                dataset.add_frame(frame, task=task)
            dataset.save_episode()

        _write_dataset_stats(target, dataset.meta.stats)
        if _uses_declared_modality(
            config=config,
            state_width=state_width,
            action_width=action_width,
        ):
            _write_split_modality(target, config=config, cameras=cameras)
        elif _uses_g1_wbc_modality(
            robot=robot,
            state_width=state_width,
            action_width=action_width,
        ):
            _write_g1_wbc_modality(target)
        logger.info("wrote %s episodes to %s", len(names), target)
        return len(names)


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.INFO, format="[dataset] %(message)s")
    if not args.input.is_file():
        print(f"error: no recording at {args.input}")
        return 1
    if args.command == "inspect":
        inspect(args.input, show_segments=args.segments)
        return 0
    if args.command == "actions":
        actions(args.input, threshold=args.threshold)
        return 0
    written = convert(
        args.input,
        args.output,
        robot=args.robot,
        repo_id=args.repo_id,
        fps=args.fps,
        video_codec=args.video_codec,
        skip_frames=args.skip_frames,
        successful_only=args.successful_only,
        task=args.task,
        g1_wbc_policy_actions=args.g1_wbc_policy_actions,
    )
    print(f"wrote {written} episodes to {args.output}")
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The observation the served checkpoint asks for, built from one read of the robot.

Pure numpy; nothing here knows about ZMQ, DDS or ROS. ``ObservationSpec`` says
which video, state, tactile and force keys the checkpoint consumes and comes
from the server's ``server_info`` (or legacy defaults for a server without it),
``CameraLayout`` holds the rig facts that are not the checkpoint's business, and
``build_observation`` turns a ``RobotSnapshot`` plus tactile inputs into the
nested dict with batch and time dims that ``get_action`` takes.

The one irregular shape is ``tactile_force.force``: it carries the batch dim
only, because the patched processor expects a bare ``(W, 6N)`` window after
unbatching, where the deform maps go through the usual ``(B, T, ...)`` path.
"""

from __future__ import annotations

import logging
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from i4h_tasks.gr00t_n17_tactile.tactile.interface import DEFORM_SHAPE, FINGERTIPS, FORCE_AXES, HAND_DOF, HANDS

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.client")

LANGUAGE_KEY = "annotation.human.task_description"
ARM_JOINTS = 7
DEFAULT_VIDEO_KEYS: tuple[str, ...] = ("high", "left_wrist_view", "right_wrist_view")
DEFAULT_STATE_GROUPS: dict[str, tuple[int, int]] = {
    "left_arm": (0, 7),
    "right_arm": (7, 14),
    "left_hand": (14, 36),
    "right_hand": (36, 58),
}
WRIST_KEYS: tuple[str, ...] = ("left_wrist_view", "right_wrist_view")
EYE_KEYS: tuple[str, ...] = ("left_eye_view", "right_eye_view")
DEFAULT_FORCE_WINDOW = 16

_DEFORM_SUFFIX = "_deform"
_WRIST_CHOICES: dict[str, tuple[str, ...]] = {
    "right": ("right_wrist_view",),
    "left": ("left_wrist_view",),
    "both": WRIST_KEYS,
    "none": (),
}


def _hand_of(name: str) -> str:
    """``left`` of ``left_thumb`` / ``left_thumb_deform`` / ``left_hand``."""
    return name.partition("_")[0]


def _wrist_keys(choice: str) -> tuple[str, ...]:
    if choice not in _WRIST_CHOICES:
        raise ValueError(f"wrist_cams must be one of {tuple(_WRIST_CHOICES)}, got {choice!r}")
    return _WRIST_CHOICES[choice]


def _tactile_keys(keys: Iterable[str]) -> tuple[str, ...]:
    """Validate ``<fingertip>_deform`` keys, keeping their order."""
    out = []
    for key in keys:
        name = str(key)
        fingertip = name[: -len(_DEFORM_SUFFIX)] if name.endswith(_DEFORM_SUFFIX) else None
        if fingertip not in FINGERTIPS:
            raise ValueError(f"unexpected tactile key {name!r}; the fingertips are {FINGERTIPS}, each with '_deform'")
        out.append(name)
    return tuple(out)


@dataclass(frozen=True)
class ObservationSpec:
    """What the served checkpoint wants in one observation. Built from server_info or defaults."""

    video_keys: tuple[str, ...]
    state_keys: tuple[str, ...]
    state_groups: dict[str, tuple[int, int]]
    tactile_keys: tuple[str, ...]
    force_window: int
    language_key: str = LANGUAGE_KEY

    @property
    def fingertips(self) -> tuple[str, ...]:
        return tuple(key[: -len(_DEFORM_SUFFIX)] for key in self.tactile_keys)

    @property
    def hands(self) -> tuple[str, ...]:
        return tuple(hand for hand in HANDS if f"{hand}_hand" in self.state_keys)

    @property
    def force_columns(self) -> int:
        return FORCE_AXES * len(self.fingertips)

    @property
    def uses_tactile(self) -> bool:
        return bool(self.tactile_keys)

    @property
    def uses_force(self) -> bool:
        return self.force_window > 0


def default_spec(
    *,
    tactile: bool = True,
    force_window: int = 0,
    hands: Iterable[str] = HANDS,
    wrist_cams: str = "both",
    head_eye: str = "left",
) -> ObservationSpec:
    """The bimanual manifest layout, for ``--check`` mode and for a server without ``server_info``."""
    chosen = set(hands)
    hand_order = tuple(hand for hand in HANDS if hand in chosen)
    video = (list(EYE_KEYS) if head_eye == "both" else ["high"]) + list(_wrist_keys(wrist_cams))
    state = ("left_arm", "right_arm") + tuple(f"{hand}_hand" for hand in hand_order)
    tactile_keys: tuple[str, ...] = ()
    if tactile:
        tactile_keys = tuple(f"{name}{_DEFORM_SUFFIX}" for name in FINGERTIPS if _hand_of(name) in chosen)
    return ObservationSpec(tuple(video), state, dict(DEFAULT_STATE_GROUPS), tactile_keys, int(force_window))


def spec_from_server_info(info: dict[str, Any]) -> ObservationSpec:
    """Read what ``server.py:served_info()`` serves.

    ``required_observation_keys`` is authoritative per modality when present; the
    legacy fields (``video_modality_keys``, ``state_groups``, ``tactile.fingertips``
    with ``checkpoint.use_tactile``) fill in for a server that lacks it.
    """
    required = info.get("required_observation_keys") or {}
    tactile_info = info.get("tactile") or {}
    checkpoint = info.get("checkpoint") or {}

    if "video" in required:
        video = tuple(str(key) for key in required["video"])
    else:
        video = tuple(str(key) for key in (info.get("video_modality_keys") or DEFAULT_VIDEO_KEYS))

    raw_groups = info.get("state_groups") or {}
    if raw_groups:
        state_groups = {str(key): (int(span[0]), int(span[1])) for key, span in raw_groups.items()}
    else:
        state_groups = dict(DEFAULT_STATE_GROUPS)
    if "state" in required:
        state = tuple(str(key) for key in required["state"])
    else:
        known = tuple(key for key in DEFAULT_STATE_GROUPS if key in state_groups)
        state = known + tuple(key for key in state_groups if key not in DEFAULT_STATE_GROUPS)

    language = required.get("language") or [LANGUAGE_KEY]
    language_key = str(language[0])

    fingertips = [str(name) for name in (tactile_info.get("fingertips") or [])]
    if "tactile" in required:
        tactile_keys = _tactile_keys(required["tactile"])
    else:
        use_tactile = checkpoint.get("use_tactile")
        if use_tactile is None:
            use_tactile = bool(fingertips)
        tactile_keys = _tactile_keys(f"{name}{_DEFORM_SUFFIX}" for name in fingertips) if use_tactile else ()

    use_force = checkpoint.get("use_tactile_force")
    if use_force is None:
        use_force = "tactile_force" in required
    force_allowed = not required or "tactile_force" in required
    force_window = 0
    if use_force and force_allowed:
        force_window = int(tactile_info.get("force_window") or 0) or DEFAULT_FORCE_WINDOW
    return ObservationSpec(video, state, state_groups, tactile_keys, force_window, language_key)


@dataclass(frozen=True)
class SpecOverrides:
    """The tri-state command-line flags that may contradict the server
    (``--checkpoint-tactile[-force]``, ``--state-hands``, ``--wrist-cams``, ``--head-eye``)."""

    checkpoint_tactile: bool | None = None
    checkpoint_tactile_force: bool | None = None
    state_hands: str = "auto"
    wrist_cams: str = "auto"
    head_eye: str | None = None


def _with_eye_keys(video: list[str]) -> list[str]:
    out: list[str] = []
    for key in video:
        if key == "high":
            out.extend(EYE_KEYS)
        else:
            out.append(key)
    return out


def _with_high(video: list[str]) -> list[str]:
    out: list[str] = []
    for key in video:
        if key in EYE_KEYS:
            if "high" not in out:
                out.append("high")
        else:
            out.append(key)
    return out


def resolve_spec(info: dict[str, Any] | None, overrides: SpecOverrides) -> tuple[ObservationSpec, list[str]]:
    """The spec the client sends, from the server's facts and the operator's explicit flags.

    An explicit flag always wins; contradicting the server only earns a
    warning (returned, logged by the caller). The one hard error is a force
    window without deform keys, which no checkpoint accepts.
    """
    warnings: list[str] = []
    if info:
        base = spec_from_server_info(info)
        server_tactile: bool | None = base.uses_tactile
        server_force: bool | None = base.uses_force
    else:
        base = default_spec(tactile=True, force_window=0)
        server_tactile = server_force = None
        warnings.append(
            "server_info unavailable; using legacy defaults (tactile on, force off, state hands both, "
            "wrist cams right) -- pass explicit flags if wrong"
        )
    video = list(base.video_keys)
    state = list(base.state_keys)
    tactile = list(base.tactile_keys)
    force_window = base.force_window
    hands = base.hands

    if overrides.checkpoint_tactile is not None:
        if server_tactile is not None and server_tactile != overrides.checkpoint_tactile:
            warnings.append(
                f"explicit --checkpoint-tactile={overrides.checkpoint_tactile} contradicts server_info "
                f"(use_tactile={server_tactile}); keeping the flag"
            )
        if not overrides.checkpoint_tactile:
            tactile = []
        elif not tactile:
            tactile = [f"{name}{_DEFORM_SUFFIX}" for name in FINGERTIPS if not hands or _hand_of(name) in hands]

    if overrides.checkpoint_tactile_force is not None:
        if server_force is not None and server_force != overrides.checkpoint_tactile_force:
            warnings.append(
                f"explicit --checkpoint-tactile-force={overrides.checkpoint_tactile_force} contradicts server_info "
                f"(use_tactile_force={server_force}); keeping the flag"
            )
        force_window = (base.force_window or DEFAULT_FORCE_WINDOW) if overrides.checkpoint_tactile_force else 0

    if force_window > 0 and not tactile:
        raise ValueError(
            "checkpoint_tactile_force requires checkpoint_tactile (a force checkpoint also needs the deform keys)"
        )

    if overrides.state_hands != "auto":
        if overrides.state_hands == "both":
            chosen: tuple[str, ...] = HANDS
        elif overrides.state_hands in HANDS:
            chosen = (overrides.state_hands,)
        else:
            raise ValueError(f"state_hands must be auto, both or one of {HANDS}, got {overrides.state_hands!r}")
        state = ["left_arm", "right_arm"] + [f"{hand}_hand" for hand in chosen]
        tactile = [key for key in tactile if _hand_of(key) in chosen]

    wrist_cams = overrides.wrist_cams
    if wrist_cams == "auto" and not info:
        wrist_cams = "right"
    if wrist_cams != "auto":
        video = [key for key in video if key not in WRIST_KEYS] + list(_wrist_keys(wrist_cams))

    if overrides.head_eye == "both":
        video = _with_eye_keys(video)
    elif overrides.head_eye in ("left", "right"):
        video = _with_high(video)
    elif overrides.head_eye is not None:
        raise ValueError(f"head_eye must be left, right or both, got {overrides.head_eye!r}")

    spec = ObservationSpec(
        tuple(video), tuple(state), dict(base.state_groups), tuple(tactile), int(force_window), base.language_key
    )
    return spec, warnings


@dataclass(frozen=True)
class CameraLayout:
    """Rig facts that are not the checkpoint's business (``--head-mode``, ``--head-eye``, ``--swap-wrist-cams``)."""

    head_mode: str = "binocular"
    head_eye: str = "left"
    swap_wrist_cams: bool = True


def resolve_layout(spec: ObservationSpec, head_mode: str, head_eye: str | None, swap_wrist_cams: bool) -> CameraLayout:
    """``head_eye`` None becomes ``both`` when the spec lists eye views, else ``left``; explicit values are kept."""
    if head_mode not in ("binocular", "mono"):
        raise ValueError(f"head_mode must be binocular or mono, got {head_mode!r}")
    if head_eye is None:
        head_eye = "both" if any(key in spec.video_keys for key in EYE_KEYS) else "left"
    return CameraLayout(head_mode=head_mode, head_eye=head_eye, swap_wrist_cams=bool(swap_wrist_cams))


@dataclass
class RobotSnapshot:
    """One read of the robot, as ``robot.py`` delivers it (BGR frames, radians).

    ``frames`` has the keys ``head``, ``left_wrist`` and ``right_wrist``, each a
    BGR uint8 ``(H, W, 3)`` frame or None; ``head`` is the stitched ``(H, 2W, 3)``
    stereo frame in binocular mode. ``arm_q`` is ``(14,)``, left arm then right,
    or None without an arm controller. ``hand_q`` maps a side to its ``(22,)``
    radians; a side that was not read is absent.
    """

    frames: dict[str, np.ndarray | None]
    arm_q: np.ndarray | None
    hand_q: dict[str, np.ndarray] = field(default_factory=dict)
    t_capture: float = 0.0


def split_head_image(stitched: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """``(H, 2W, 3)`` to ``(left_eye, right_eye)`` views."""
    width = stitched.shape[1] // 2
    return stitched[:, :width, :], stitched[:, width:, :]


def to_rgb(bgr: np.ndarray) -> np.ndarray:
    """BGR to RGB as a C-contiguous copy."""
    return np.ascontiguousarray(bgr[:, :, ::-1])


def select_video(
    frames: dict[str, np.ndarray | None], layout: CameraLayout, video_keys: Iterable[str]
) -> dict[str, np.ndarray]:
    """The RGB frames for ``video_keys``, routed per ``layout``.

    The wrist swap happens on the frames, before any key is assigned, so it
    stays right when one of them is missing. A wanted key whose frame is
    missing is simply absent; the caller decides whether that is fatal.
    """
    wanted = set(video_keys)
    head = frames.get("head")
    left_wrist = frames.get("left_wrist")
    right_wrist = frames.get("right_wrist")
    if layout.swap_wrist_cams:
        left_wrist, right_wrist = right_wrist, left_wrist

    video: dict[str, np.ndarray] = {}
    if head is not None:
        if layout.head_mode == "mono":
            video["high"] = to_rgb(head)
        else:
            left_eye, right_eye = split_head_image(head)
            if layout.head_eye == "left":
                video["high"] = to_rgb(left_eye)
            elif layout.head_eye == "right":
                video["high"] = to_rgb(right_eye)
            else:
                video["left_eye_view"] = to_rgb(left_eye)
                video["right_eye_view"] = to_rgb(right_eye)
    if left_wrist is not None:
        video["left_wrist_view"] = to_rgb(left_wrist)
    if right_wrist is not None:
        video["right_wrist_view"] = to_rgb(right_wrist)
    return {key: value for key, value in video.items() if key in wanted}


def select_state(snapshot: RobotSnapshot, spec: ObservationSpec) -> dict[str, np.ndarray]:
    """float32 state vectors for ``spec.state_keys``, in spec order; zeros for what was not read."""
    if snapshot.arm_q is None:
        arm_q = np.zeros(2 * ARM_JOINTS, dtype=np.float32)
    else:
        arm_q = np.asarray(snapshot.arm_q, dtype=np.float32).reshape(-1)
    state: dict[str, np.ndarray] = {}
    for key in spec.state_keys:
        if key == "left_arm":
            state[key] = arm_q[:ARM_JOINTS].copy()
        elif key == "right_arm":
            state[key] = arm_q[ARM_JOINTS : 2 * ARM_JOINTS].copy()
        elif key.endswith("_hand"):
            value = snapshot.hand_q.get(_hand_of(key))
            if value is None:
                state[key] = np.zeros(HAND_DOF, dtype=np.float32)
            else:
                state[key] = np.asarray(value, dtype=np.float32).reshape(-1)
        else:
            raise ValueError(f"unknown state key {key!r}")
    return state


@dataclass
class TactileInputs:
    """Deform maps per tactile key and the force window, as ``build_observation`` consumes them."""

    deform: dict[str, np.ndarray]
    force: np.ndarray | None
    complete: bool


def read_tactile(source: Any | None, spec: ObservationSpec) -> TactileInputs:
    """Tactile inputs for ``spec`` from a source built for exactly ``spec.fingertips``.

    Without a source the maps are zeros. While the source has no complete frame
    the fingertips that do report keep their live maps (``deform_maps()``, the
    old client's per-finger substitution) and only the silent ones are zeros;
    ``complete`` is then False and the caller warns. The force window is
    ``(W, 6N)`` with column ``6*i + a`` holding axis ``a`` of fingertip ``i``, the
    layout the dataset's ``observation.tactile.force`` uses.
    """
    keys = spec.tactile_keys
    frame = source.latest() if source is not None else None
    if frame is not None:
        maps = np.asarray(frame.deform)
        if maps.shape[0] != len(keys):
            raise ValueError(f"tactile source delivers {maps.shape[0]} fingertips, the spec wants {len(keys)}")
        deform = {key: maps[index] for index, key in enumerate(keys)}
    else:
        partial = source.deform_maps() if hasattr(source, "deform_maps") else {}
        deform = {}
        for key, fingertip in zip(keys, spec.fingertips):
            value = partial.get(fingertip)
            deform[key] = np.zeros(DEFORM_SHAPE, dtype=np.uint8) if value is None else np.asarray(value)

    force = None
    if spec.uses_force:
        width = spec.force_window
        columns = spec.force_columns
        if source is not None:
            window = np.asarray(source.force_window(width), dtype=np.float32)
            if window.shape != (len(spec.fingertips), width, FORCE_AXES):
                raise ValueError(
                    f"force window is {window.shape}, expected {(len(spec.fingertips), width, FORCE_AXES)}"
                )
            force = np.ascontiguousarray(window.transpose(1, 0, 2).reshape(width, columns))
        else:
            force = np.zeros((width, columns), dtype=np.float32)
    complete = frame is not None if keys else True
    return TactileInputs(deform=deform, force=force, complete=complete)


def read_contact_metric(source: Any | None, fingertips: Iterable[str], hand: str = "left") -> float:
    """Max |force| over the six axes of one hand's fingertips in the newest row; 0.0 without data.

    A cheap contact proxy for the same-plan refine gate: about zero in free
    space, rising on contact. The row comes from the subscriber's force sampler
    (30 Hz, the dataset rate), so it can trail the newest message by one period.
    """
    if source is None:
        return 0.0
    names = tuple(fingertips)
    picked = [index for index, name in enumerate(names) if name.startswith(f"{hand}_")]
    if not picked:
        return 0.0
    try:
        newest = np.asarray(source.force_window(1), dtype=np.float32)[:, -1, :]
        return float(np.abs(newest[picked]).max())
    except Exception:
        return 0.0


def add_batch_time_dims(obs: dict[str, Any]) -> dict[str, Any]:
    """``(B=1, T=1)`` in front of every array; strings become ``[[str]]``; dicts recurse."""
    out: dict[str, Any] = {}
    for key, value in obs.items():
        if isinstance(value, np.ndarray):
            out[key] = value[np.newaxis, np.newaxis, ...]
        elif isinstance(value, dict):
            out[key] = add_batch_time_dims(value)
        else:
            out[key] = [[value]]
    return out


def build_observation(
    snapshot: RobotSnapshot,
    spec: ObservationSpec,
    layout: CameraLayout,
    task: str,
    tactile: TactileInputs | None,
) -> dict[str, Any]:
    """The nested GR00T observation with batch/time dims (batch dim only on ``tactile_force.force``)."""
    obs: dict[str, Any] = {
        "video": select_video(snapshot.frames, layout, spec.video_keys),
        "state": select_state(snapshot, spec),
        "language": {spec.language_key: task},
    }
    if spec.uses_tactile:
        if tactile is None:
            raise ValueError("the spec has tactile keys but no tactile inputs were given")
        obs["tactile"] = dict(tactile.deform)
    out = add_batch_time_dims(obs)
    if spec.uses_force:
        if tactile is None or tactile.force is None:
            raise ValueError("the spec has a force window but no force inputs were given")
        out["tactile_force"] = {"force": np.asarray(tactile.force, dtype=np.float32)[np.newaxis, ...]}
    return out


def describe_observation(obs: dict[str, Any]) -> list[str]:
    """One line per leaf, for ``--check``: shape, dtype and range of every array, the value of anything else."""
    lines: list[str] = []

    def walk(node: dict[str, Any], prefix: str) -> None:
        for key, value in node.items():
            path = f"{prefix}.{key}" if prefix else str(key)
            if isinstance(value, dict):
                walk(value, path)
            elif isinstance(value, np.ndarray):
                stats = f" min={value.min():.4f} max={value.max():.4f} mean={value.mean():.4f}" if value.size else ""
                lines.append(f"{path:45s} shape={str(value.shape):20s} dtype={value.dtype}{stats}")
            else:
                lines.append(f"{path:45s} value={value!r}")

    walk(obs, "")
    return lines

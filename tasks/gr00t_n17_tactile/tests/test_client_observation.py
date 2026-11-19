# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The observation builder sends exactly what the served checkpoint asks for.

Camera routing, the hand topology and the force-window layout are the places a
deployment goes wrong silently, so each is pinned on synthetic frames whose
fill values tell the streams apart. Pure numpy; runs in the light venv.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.client import observation as obs_mod  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.client.observation import (  # noqa: E402
    EYE_KEYS,
    CameraLayout,
    RobotSnapshot,
    SpecOverrides,
    TactileInputs,
    add_batch_time_dims,
    build_observation,
    default_spec,
    describe_observation,
    read_contact_metric,
    read_tactile,
    resolve_layout,
    resolve_spec,
    select_state,
    select_video,
    spec_from_server_info,
    split_head_image,
    to_rgb,
)
from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS, TactileFrame  # noqa: E402

LEFT = FINGERTIPS[:5]
ALL_STATE = ("left_arm", "right_arm", "left_hand", "right_hand")


def _served(bimanual: bool = True) -> dict:
    """What ``server.py:served_info()`` returns for the manifest checkpoint and for a left-only finetune."""
    fingertips = list(FINGERTIPS) if bimanual else list(LEFT)
    video = ["high", "left_wrist_view", "right_wrist_view"] if bimanual else ["high", "right_wrist_view"]
    state = list(ALL_STATE) if bimanual else ["left_arm", "right_arm", "left_hand"]
    groups = {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 36], "right_hand": [36, 58]}
    return {
        "task": "gr00t_n17_tactile/h2_sharpa_tactile",
        "embodiment": "h2_sharpa",
        "video_modality_keys": video,
        "state_groups": {key: groups[key] for key in state},
        "action_horizon": 32,
        "execution_steps": 16,
        "control_hz": 30.0,
        "tactile": {
            "enabled": True,
            "source": "live",
            "fingertips": fingertips,
            "num_tokens": len(fingertips),
            "feature_dim": 512,
            "force_window": 16 if bimanual else 0,
            "delay_aug": bimanual,
            "refresh": "per-step",
        },
        "cascade": {"enabled": False, "fast_loops": 2},
        "checkpoint": {
            "class": "Gr00tN1d7Tactile",
            "use_tactile": True,
            "use_tactile_force": bimanual,
            "delay_aug": bimanual,
            "tau_split": 0.5,
            "force_delay_max": 16 if bimanual else 0,
            "num_tactile_tokens": len(fingertips),
        },
        "required_observation_keys": {
            "video": video,
            "state": state,
            "language": ["annotation.human.task_description"],
            "tactile": [f"{name}_deform" for name in fingertips],
            # The modality config always declares the force category; the checkpoint decides whether it is used.
            "tactile_force": ["force"],
        },
    }


# -- spec from server_info ---------------------------------------------------


def test_the_bimanual_checkpoint_wants_everything() -> None:
    spec = spec_from_server_info(_served())
    assert spec.video_keys == ("high", "left_wrist_view", "right_wrist_view")
    assert spec.state_keys == ALL_STATE
    assert spec.state_groups == {
        "left_arm": (0, 7),
        "right_arm": (7, 14),
        "left_hand": (14, 36),
        "right_hand": (36, 58),
    }
    assert spec.tactile_keys == tuple(f"{name}_deform" for name in FINGERTIPS)
    assert spec.fingertips == FINGERTIPS and spec.hands == ("left", "right")
    assert spec.force_window == 16 and spec.force_columns == 60
    assert spec.uses_tactile and spec.uses_force
    assert spec.language_key == "annotation.human.task_description"


def test_a_left_only_checkpoint_wants_one_hand_and_no_force() -> None:
    spec = spec_from_server_info(_served(bimanual=False))
    assert spec.video_keys == ("high", "right_wrist_view")
    assert spec.state_keys == ("left_arm", "right_arm", "left_hand") and spec.hands == ("left",)
    assert spec.fingertips == LEFT and spec.force_columns == 30
    assert spec.force_window == 0 and not spec.uses_force and spec.uses_tactile
    assert list(spec.state_groups) == ["left_arm", "right_arm", "left_hand"]


def test_legacy_fields_fill_in_without_required_keys() -> None:
    info = _served()
    del info["required_observation_keys"]
    info["state_groups"] = {"right_hand": [36, 58], "left_arm": [0, 7], "right_arm": [7, 14]}
    spec = spec_from_server_info(info)
    assert spec.video_keys == ("high", "left_wrist_view", "right_wrist_view")
    assert spec.state_keys == ("left_arm", "right_arm", "right_hand"), "canonical order, not served order"
    assert spec.hands == ("right",)
    assert spec.tactile_keys == tuple(f"{name}_deform" for name in FINGERTIPS)
    assert spec.force_window == 16

    info["checkpoint"]["use_tactile"] = False
    info["checkpoint"]["use_tactile_force"] = False
    vision_only = spec_from_server_info(info)
    assert vision_only.tactile_keys == () and vision_only.force_window == 0

    del info["checkpoint"]
    assert spec_from_server_info(info).uses_tactile, "no checkpoint facts: fingertips alone mean tactile"


def test_a_force_checkpoint_without_a_window_gets_the_default() -> None:
    info = _served()
    info["tactile"]["force_window"] = 0
    assert spec_from_server_info(info).force_window == obs_mod.DEFAULT_FORCE_WINDOW


def test_an_unknown_tactile_key_is_rejected() -> None:
    info = _served()
    info["required_observation_keys"]["tactile"][0] = "left_palm_deform"
    with pytest.raises(ValueError, match="left_palm_deform"):
        spec_from_server_info(info)
    info["required_observation_keys"]["tactile"][0] = "left_thumb"
    with pytest.raises(ValueError, match="_deform"):
        spec_from_server_info(info)


def test_default_spec_is_the_manifest_layout() -> None:
    spec = default_spec()
    assert spec.video_keys == ("high", "left_wrist_view", "right_wrist_view")
    assert spec.state_keys == ALL_STATE and spec.fingertips == FINGERTIPS and spec.force_window == 0
    left = default_spec(hands=("left",), wrist_cams="right", head_eye="both", force_window=16)
    assert left.video_keys == EYE_KEYS + ("right_wrist_view",)
    assert left.state_keys == ("left_arm", "right_arm", "left_hand") and left.fingertips == LEFT
    assert left.force_window == 16
    assert default_spec(tactile=False).tactile_keys == ()


# -- resolve_spec --------------------------------------------------------------


def test_no_server_info_means_the_legacy_defaults_and_a_warning() -> None:
    spec, warnings = resolve_spec(None, SpecOverrides())
    assert spec.video_keys == ("high", "right_wrist_view"), "legacy single-left-arm wrist view"
    assert spec.state_keys == ALL_STATE
    assert spec.fingertips == FINGERTIPS and spec.force_window == 0
    assert len(warnings) == 1 and "server_info unavailable" in warnings[0]


def test_the_server_wins_when_nothing_is_explicit() -> None:
    spec, warnings = resolve_spec(_served(bimanual=False), SpecOverrides())
    assert spec == spec_from_server_info(_served(bimanual=False)) and warnings == []


def test_explicit_no_tactile_drops_the_keys_and_warns() -> None:
    spec, warnings = resolve_spec(_served(bimanual=False), SpecOverrides(checkpoint_tactile=False))
    assert spec.tactile_keys == () and not spec.uses_force
    assert any("--checkpoint-tactile=False" in w and "use_tactile=True" in w for w in warnings)


def test_explicit_tactile_on_a_vision_only_server_adds_the_hands_fingertips() -> None:
    info = _served(bimanual=False)
    info["checkpoint"]["use_tactile"] = False
    info["required_observation_keys"].pop("tactile")
    info["required_observation_keys"].pop("tactile_force")
    spec, warnings = resolve_spec(info, SpecOverrides(checkpoint_tactile=True))
    assert spec.fingertips == LEFT, "only the hand the server has state for"
    assert any("use_tactile=False" in w for w in warnings)


def test_force_without_tactile_is_an_error() -> None:
    with pytest.raises(ValueError, match="checkpoint_tactile_force requires checkpoint_tactile"):
        resolve_spec(_served(), SpecOverrides(checkpoint_tactile=False))
    with pytest.raises(ValueError, match="checkpoint_tactile_force requires checkpoint_tactile"):
        resolve_spec(None, SpecOverrides(checkpoint_tactile=False, checkpoint_tactile_force=True))
    spec, _ = resolve_spec(_served(), SpecOverrides(checkpoint_tactile=False, checkpoint_tactile_force=False))
    assert spec.tactile_keys == () and spec.force_window == 0


def test_explicit_force_flags_override_the_server() -> None:
    off, warnings = resolve_spec(_served(), SpecOverrides(checkpoint_tactile_force=False))
    assert off.force_window == 0 and any("--checkpoint-tactile-force=False" in w for w in warnings)
    on, warnings = resolve_spec(_served(bimanual=False), SpecOverrides(checkpoint_tactile_force=True))
    assert on.force_window == obs_mod.DEFAULT_FORCE_WINDOW and any("use_tactile_force=False" in w for w in warnings)


def test_state_hands_left_restricts_state_and_fingertips() -> None:
    spec, _ = resolve_spec(_served(), SpecOverrides(state_hands="left"))
    assert spec.state_keys == ("left_arm", "right_arm", "left_hand") and spec.hands == ("left",)
    assert spec.fingertips == LEFT and spec.force_columns == 30 and spec.force_window == 16
    both, _ = resolve_spec(_served(bimanual=False), SpecOverrides(state_hands="both"))
    assert both.state_keys == ALL_STATE and both.fingertips == LEFT, "both hands' state, the server's fingertips"


def test_wrist_cams_override_replaces_the_wrist_keys_only() -> None:
    none, _ = resolve_spec(_served(), SpecOverrides(wrist_cams="none"))
    assert none.video_keys == ("high",)
    left, _ = resolve_spec(_served(), SpecOverrides(wrist_cams="left"))
    assert left.video_keys == ("high", "left_wrist_view")
    both, _ = resolve_spec(_served(bimanual=False), SpecOverrides(wrist_cams="both"))
    assert both.video_keys == ("high", "left_wrist_view", "right_wrist_view")


def test_head_eye_override_swaps_the_head_keys() -> None:
    both, _ = resolve_spec(_served(bimanual=False), SpecOverrides(head_eye="both"))
    assert both.video_keys == EYE_KEYS + ("right_wrist_view",)
    info = _served(bimanual=False)
    info["required_observation_keys"]["video"] = list(EYE_KEYS) + ["right_wrist_view"]
    kept, _ = resolve_spec(info, SpecOverrides())
    assert kept.video_keys == EYE_KEYS + ("right_wrist_view",)
    left, _ = resolve_spec(info, SpecOverrides(head_eye="left"))
    assert left.video_keys == ("high", "right_wrist_view")


def test_resolve_layout_picks_the_eye_from_the_spec() -> None:
    spec = spec_from_server_info(_served())
    assert resolve_layout(spec, "binocular", None, True) == CameraLayout("binocular", "left", True)
    assert resolve_layout(spec, "mono", "right", False) == CameraLayout("mono", "right", False)
    eyes, _ = resolve_spec(_served(), SpecOverrides(head_eye="both"))
    assert resolve_layout(eyes, "binocular", None, True).head_eye == "both"
    with pytest.raises(ValueError, match="head_mode"):
        resolve_layout(spec, "stereo", None, True)


# -- video -----------------------------------------------------------------------


def _frames(head: bool = True, left: bool = True, right: bool = True) -> dict[str, np.ndarray | None]:
    stitched = np.zeros((4, 6, 3), dtype=np.uint8)
    stitched[:, :3] = 30
    stitched[:, 3:] = 40
    stitched[0, 0] = (1, 2, 3)
    return {
        "head": stitched if head else None,
        "left_wrist": np.full((4, 3, 3), 10, dtype=np.uint8) if left else None,
        "right_wrist": np.full((4, 3, 3), 20, dtype=np.uint8) if right else None,
    }


def test_split_and_rgb_are_the_old_clients() -> None:
    left, right = split_head_image(_frames()["head"])
    assert left.shape == right.shape == (4, 3, 3) and left[1, 1, 0] == 30 and right[1, 1, 0] == 40
    rgb = to_rgb(_frames()["head"])
    assert tuple(rgb[0, 0]) == (3, 2, 1) and rgb.flags["C_CONTIGUOUS"]


def test_binocular_head_routes_the_chosen_eye_to_high() -> None:
    keys = ("high", "left_wrist_view", "right_wrist_view")
    left = select_video(_frames(), CameraLayout("binocular", "left", False), keys)
    assert left["high"].shape == (4, 3, 3) and left["high"][1, 1, 0] == 30 and tuple(left["high"][0, 0]) == (3, 2, 1)
    right = select_video(_frames(), CameraLayout("binocular", "right", False), keys)
    assert right["high"][1, 1, 0] == 40
    both = select_video(_frames(), CameraLayout("binocular", "both", False), EYE_KEYS)
    assert set(both) == set(EYE_KEYS)
    assert both["left_eye_view"][1, 1, 0] == 30 and both["right_eye_view"][1, 1, 0] == 40
    mono = select_video(_frames(), CameraLayout("mono", "left", False), ("high",))
    assert mono["high"].shape == (4, 6, 3), "mono sends the whole frame"


def test_the_wrist_swap_happens_on_the_frames() -> None:
    keys = ("left_wrist_view", "right_wrist_view")
    swapped = select_video(_frames(), CameraLayout(swap_wrist_cams=True), keys)
    assert swapped["right_wrist_view"][0, 0, 0] == 10, "right_wrist_view shows the left-wrist camera"
    assert swapped["left_wrist_view"][0, 0, 0] == 20
    identity = select_video(_frames(), CameraLayout(swap_wrist_cams=False), keys)
    assert identity["right_wrist_view"][0, 0, 0] == 20 and identity["left_wrist_view"][0, 0, 0] == 10
    one_missing = select_video(_frames(left=False), CameraLayout(swap_wrist_cams=True), keys)
    assert set(one_missing) == {"left_wrist_view"} and one_missing["left_wrist_view"][0, 0, 0] == 20


def test_video_is_pruned_to_the_wanted_keys_and_missing_frames_are_absent() -> None:
    video = select_video(_frames(), CameraLayout(), ("high", "right_wrist_view"))
    assert set(video) == {"high", "right_wrist_view"}
    assert set(select_video(_frames(head=False), CameraLayout(), ("high", "right_wrist_view"))) == {"right_wrist_view"}
    assert select_video({"head": None, "left_wrist": None, "right_wrist": None}, CameraLayout(), ("high",)) == {}


# -- state -----------------------------------------------------------------------


def test_state_follows_the_spec_order_and_zero_fills() -> None:
    spec = spec_from_server_info(_served())
    arm = np.arange(14, dtype=np.float64) * 0.1
    snapshot = RobotSnapshot(frames={}, arm_q=arm, hand_q={"left": np.full(22, 0.5, dtype=np.float32)}, t_capture=0.0)
    state = select_state(snapshot, spec)
    assert list(state) == list(ALL_STATE)
    np.testing.assert_allclose(state["left_arm"], arm[:7].astype(np.float32))
    np.testing.assert_allclose(state["right_arm"], arm[7:].astype(np.float32))
    assert state["left_hand"][0] == np.float32(0.5) and not state["right_hand"].any()
    assert all(value.dtype == np.float32 for value in state.values())
    no_arm = select_state(RobotSnapshot(frames={}, arm_q=None), spec_from_server_info(_served(bimanual=False)))
    assert list(no_arm) == ["left_arm", "right_arm", "left_hand"]
    assert not no_arm["left_arm"].any() and no_arm["right_arm"].shape == (7,) and no_arm["left_hand"].shape == (22,)


# -- tactile ---------------------------------------------------------------------


class _FakeSource:
    def __init__(self, fingertips, window: int, frame: bool = True, silent: tuple[str, ...] | None = None) -> None:
        self.fingertips = tuple(fingertips)
        self.window = window
        self.has_frame = frame
        # Without a complete frame every fingertip is silent unless the test names the silent ones.
        self.silent = self.fingertips if silent is None and not frame else tuple(silent or ())
        self.calls: list[int] = []

    def deform_maps(self):
        return {
            name: None if name in self.silent else np.full((240, 240), i + 1, dtype=np.uint8)
            for i, name in enumerate(self.fingertips)
        }

    def latest(self):
        if not self.has_frame:
            return None
        n = len(self.fingertips)
        deform = np.stack([np.full((240, 240), i + 1, dtype=np.uint8) for i in range(n)])
        return TactileFrame(
            timestamp=1.0, deform=deform, force=np.zeros((n, 6), np.float32), fingertips=self.fingertips
        )

    def force_window(self, length: int) -> np.ndarray:
        self.calls.append(length)
        n = len(self.fingertips)
        return np.arange(n * length * 6, dtype=np.float32).reshape(n, length, 6)


def test_read_tactile_maps_each_key_to_its_fingertip_and_lays_out_force_columns() -> None:
    spec = spec_from_server_info(_served())
    source = _FakeSource(spec.fingertips, 16)
    inputs = read_tactile(source, spec)
    assert list(inputs.deform) == list(spec.tactile_keys) and inputs.complete
    for i, key in enumerate(spec.tactile_keys):
        assert inputs.deform[key].shape == (240, 240) and inputs.deform[key][0, 0] == i + 1
    assert inputs.force.shape == (16, 60) and inputs.force.dtype == np.float32 and source.calls == [16]
    window = source.force_window(16)
    for i in range(10):
        for a in range(6):
            np.testing.assert_array_equal(inputs.force[:, 6 * i + a], window[i, :, a])


def test_read_tactile_zero_fills_an_incomplete_stream_or_no_source() -> None:
    spec = spec_from_server_info(_served())
    inputs = read_tactile(_FakeSource(spec.fingertips, 16, frame=False), spec)
    assert not inputs.complete and set(inputs.deform) == set(spec.tactile_keys)
    assert all(not value.any() and value.dtype == np.uint8 for value in inputs.deform.values())
    assert inputs.force.shape == (16, 60), "the force history is served even while a map is missing"
    none = read_tactile(None, spec)
    assert not none.complete and not none.force.any() and none.force.shape == (16, 60)
    left = read_tactile(None, spec_from_server_info(_served(bimanual=False)))
    assert len(left.deform) == 5 and left.force is None
    vision, _ = resolve_spec(_served(), SpecOverrides(checkpoint_tactile=False, checkpoint_tactile_force=False))
    empty = read_tactile(None, vision)
    assert empty.deform == {} and empty.force is None and empty.complete


def test_read_tactile_zero_fills_only_the_silent_fingertips() -> None:
    """The old client substituted zeros per finger; one dead sensor leaves the other nine maps live."""
    spec = spec_from_server_info(_served())
    inputs = read_tactile(_FakeSource(spec.fingertips, 16, frame=False, silent=("left_pinky",)), spec)
    assert not inputs.complete and list(inputs.deform) == list(spec.tactile_keys)
    assert not inputs.deform["left_pinky_deform"].any()
    for i, key in enumerate(spec.tactile_keys):
        if key != "left_pinky_deform":
            assert inputs.deform[key][0, 0] == i + 1 and inputs.deform[key].dtype == np.uint8

    class _NoPartial:  # a source without the per-fingertip view still zero-fills everything
        def latest(self):
            return None

        def force_window(self, length: int):
            return np.zeros((10, length, 6), dtype=np.float32)

    inputs = read_tactile(_NoPartial(), spec)
    assert not inputs.complete and all(not value.any() for value in inputs.deform.values())


def test_contact_metric_reads_one_hand_in_the_newest_row() -> None:
    class _Source:
        def force_window(self, length: int) -> np.ndarray:
            assert length == 1, "the gate reads the newest row only"
            window = np.zeros((10, length, 6), dtype=np.float32)
            window[2, -1, 1] = -3.5  # left middle
            window[7, -1, 2] = 8.0  # right middle
            return window

    assert read_contact_metric(_Source(), FINGERTIPS, "left") == pytest.approx(3.5)
    assert read_contact_metric(_Source(), FINGERTIPS, "right") == pytest.approx(8.0)
    assert read_contact_metric(None, FINGERTIPS) == 0.0
    assert read_contact_metric(_Source(), FINGERTIPS[5:], "left") == 0.0

    class _Broken:
        def force_window(self, length: int):
            raise RuntimeError("stream closed")

    assert read_contact_metric(_Broken(), FINGERTIPS) == 0.0


# -- the assembled observation -------------------------------------------------


def test_batch_time_dims_wrap_arrays_and_strings() -> None:
    out = add_batch_time_dims({"video": {"high": np.zeros((4, 3, 3), np.uint8)}, "language": {"k": "task"}, "n": 3})
    assert out["video"]["high"].shape == (1, 1, 4, 3, 3) and out["language"]["k"] == [["task"]] and out["n"] == [[3]]


def test_build_observation_has_the_served_shapes() -> None:
    spec = spec_from_server_info(_served())
    layout = resolve_layout(spec, "binocular", None, True)
    snapshot = RobotSnapshot(frames=_frames(), arm_q=np.zeros(14), hand_q={}, t_capture=0.0)
    tactile = read_tactile(_FakeSource(spec.fingertips, 16), spec)
    obs = build_observation(snapshot, spec, layout, "fold the pad", tactile)
    assert set(obs) == {"video", "state", "language", "tactile", "tactile_force"}
    assert set(obs["video"]) == set(spec.video_keys)
    assert obs["video"]["high"].shape == (1, 1, 4, 3, 3) and obs["video"]["high"].dtype == np.uint8
    assert obs["video"]["right_wrist_view"][0, 0, 0, 0, 0] == 10, "swap: right_wrist_view is the left-wrist camera"
    assert {key: value.shape for key, value in obs["state"].items()} == {
        "left_arm": (1, 1, 7),
        "right_arm": (1, 1, 7),
        "left_hand": (1, 1, 22),
        "right_hand": (1, 1, 22),
    }
    assert obs["language"] == {"annotation.human.task_description": [["fold the pad"]]}
    assert list(obs["tactile"]) == list(spec.tactile_keys)
    assert all(value.shape == (1, 1, 240, 240) and value.dtype == np.uint8 for value in obs["tactile"].values())
    assert obs["tactile_force"]["force"].shape == (1, 16, 60), "batch dim only"
    assert obs["tactile_force"]["force"].dtype == np.float32
    assert len(describe_observation(obs)) == 3 + 4 + 1 + 10 + 1
    assert describe_observation(obs)[0].startswith("video.high")
    assert any("dtype=float32" in line and "tactile_force.force" in line for line in describe_observation(obs))


def test_a_vision_only_spec_sends_no_tactile_keys() -> None:
    spec, _ = resolve_spec(_served(), SpecOverrides(checkpoint_tactile=False, checkpoint_tactile_force=False))
    snapshot = RobotSnapshot(frames=_frames(), arm_q=None, hand_q={}, t_capture=0.0)
    obs = build_observation(snapshot, spec, CameraLayout(), "t", None)
    assert set(obs) == {"video", "state", "language"}
    with pytest.raises(ValueError, match="tactile"):
        build_observation(snapshot, spec_from_server_info(_served()), CameraLayout(), "t", None)
    left = spec_from_server_info(_served(bimanual=False))
    partial = build_observation(
        snapshot,
        left,
        CameraLayout(),
        "t",
        TactileInputs({k: np.zeros((240, 240), np.uint8) for k in left.tactile_keys}, None, True),
    )
    assert "tactile_force" not in partial and len(partial["tactile"]) == 5

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""What the annotator shows the VLM, and how it turns replies into labels (NVBug 6941887)."""

from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import h5py
import numpy as np
import openai
import pytest

from i4h_common.taskdef import ANNOTATE_JUDGES
from i4h_tools.annotator import cli

#: The prompt every placement task has been graded with; scissor pick-and-place
#: passes 6/6 against it, so the default must not drift.
SHIPPED_PROMPT = (
    "You are grading a robot manipulation attempt. The images are evenly spaced frames "
    "from a single attempt, in order. Judge completion primarily from the visible terminal "
    "state in the final images; use earlier images as motion context. Successful placement "
    "does not require the robot to keep holding the object after releasing it at the target.\n\n"
    "Task: {task}\n\n"
    "Answer with exactly one word — SUCCESS if the task was completed, FAILURE otherwise — "
    "then a newline and one short sentence of justification."
)


class _FakeVLM:
    """Stands in for ``openai.OpenAI``; answers FAILURE for demos whose frames are dark."""

    calls: list[list[dict]] = []

    def __init__(self, **_kwargs) -> None:
        self.chat = SimpleNamespace(completions=SimpleNamespace(create=self._create))

    def _create(self, *, model, messages, max_tokens, temperature):
        assert temperature == 0.0
        content = messages[0]["content"]
        type(self).calls.append(content)
        verdict = "FAILURE\nThe probe never reaches the phantom." if _dark(content) else "SUCCESS\nThe probe scans."
        return SimpleNamespace(choices=[SimpleNamespace(message=SimpleNamespace(content=verdict))])


def _dark(content: list[dict]) -> bool:
    import base64
    import io

    from PIL import Image

    image = next(item for item in content if item["type"] == "image_url")
    payload = image["image_url"]["url"].split(",", 1)[1]
    return np.asarray(Image.open(io.BytesIO(base64.b64decode(payload)))).mean() < 64


@pytest.fixture
def vlm(monkeypatch: pytest.MonkeyPatch) -> type[_FakeVLM]:
    _FakeVLM.calls = []
    monkeypatch.setattr(openai, "OpenAI", _FakeVLM)
    return _FakeVLM


def _video(frames: int, value: int) -> np.ndarray:
    video = np.full((frames, 8, 8, 3), value, dtype=np.uint8)
    video[:, 0, 0, 0] = np.arange(frames)  # distinct frames, so sampling is observable
    return video


def _recording(path: Path, demos: dict[str, dict[str, np.ndarray]]) -> Path:
    with h5py.File(path, "w") as handle:
        data = handle.create_group("data")
        for name, cameras in demos.items():
            demo = data.create_group(name)
            frames = len(next(iter(cameras.values())))
            demo.create_dataset("actions", data=np.zeros((frames, 6), dtype=np.float32))
            demo.attrs["success"] = True
            for camera, video in cameras.items():
                demo.create_dataset(f"obs/{camera}", data=video)
        data.attrs["total"] = len(demos)
    return path


def test_the_default_prompt_is_the_shipped_placement_prompt() -> None:
    assert cli.PROMPT == SHIPPED_PROMPT
    assert cli.build_prompt("Put it in the tray") == SHIPPED_PROMPT.format(task="Put it in the tray")


def test_the_trajectory_judge_does_not_ask_for_a_terminal_state() -> None:
    prompt = cli.build_prompt("Scan the phantom", judge="trajectory", cameras=2)

    assert "terminal state" not in prompt
    assert "visibly happens across the images" in prompt
    assert "2 cameras" in prompt
    assert prompt.endswith(SHIPPED_PROMPT.split("{task}")[1])
    assert "Task: Scan the phantom" in prompt


def test_the_annotator_knows_every_judge_a_manifest_may_name() -> None:
    assert set(cli.JUDGES) == set(ANNOTATE_JUDGES)
    with pytest.raises(ValueError, match="unknown judge"):
        cli.build_prompt("x", judge="final")


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, []),
        ("wrist", ["wrist"]),
        ("room,wrist", ["room", "wrist"]),
        (["room", "wrist"], ["room", "wrist"]),
        (["room, wrist", "head"], ["room", "wrist", "head"]),
    ],
)
def test_camera_list(value, expected) -> None:
    assert cli.camera_list(value) == expected


def test_one_camera_is_sent_without_labels_and_several_with_them() -> None:
    frame = np.zeros((4, 4, 3), dtype=np.uint8)

    single = cli.build_content("t", {"room": [frame, frame]})
    double = cli.build_content("t", {"room": [frame], "wrist": [frame]}, judge="trajectory")

    assert [item["type"] for item in single] == ["text", "image_url", "image_url"]
    assert single[0]["text"] == SHIPPED_PROMPT.format(task="t")
    assert [item.get("text") for item in double if item["type"] == "text"][1:] == ["Camera room:", "Camera wrist:"]
    assert [item["type"] for item in double] == ["text", "text", "image_url", "text", "image_url"]


def test_without_camera_the_first_by_name_is_graded(tmp_path, vlm) -> None:
    # Only the room camera is bright, so a SUCCESS proves it was the one graded.
    path = _recording(tmp_path / "r.hdf5", {"demo_0": {"wrist": _video(20, 10), "room": _video(20, 200)}})

    results = cli.annotate(path, task="Put it in the tray")

    assert results == [("demo_0", True, "The probe scans.")]
    (content,) = vlm.calls
    assert content[0]["text"] == SHIPPED_PROMPT.format(task="Put it in the tray")
    assert sum(item["type"] == "image_url" for item in content) == 6


def test_several_cameras_are_sampled_at_the_same_instants(tmp_path, vlm) -> None:
    path = _recording(tmp_path / "r.hdf5", {"demo_0": {"room": _video(30, 200), "wrist": _video(30, 200)}})

    cli.annotate(path, task="Scan", camera="room,wrist", judge="trajectory", frames=4)

    (content,) = vlm.calls
    assert sum(item["type"] == "image_url" for item in content) == 8
    assert "visibly happens across the images" in content[0]["text"]


def test_a_failed_episode_is_still_labelled_failure_and_filtered_out(tmp_path, vlm) -> None:
    path = _recording(
        tmp_path / "r.hdf5",
        {"demo_0": {"wrist": _video(20, 200)}, "demo_1": {"wrist": _video(20, 10)}},
    )

    results = cli.annotate(path, task="Scan", camera="wrist", write=True)
    kept = cli.filter_successful(path, tmp_path / "filtered.hdf5", results)

    assert [(name, success) for name, success, _ in results] == [("demo_0", True), ("demo_1", False)]
    assert kept == 1
    with h5py.File(path, "r") as handle:
        assert not handle["data/demo_1"].attrs["vlm_success"]
        assert handle["data/demo_1"].attrs["vlm_success_reason"] == "The probe never reaches the phantom."


def test_all_failures_refuse_to_write_an_empty_filter(tmp_path, vlm) -> None:
    path = _recording(tmp_path / "r.hdf5", {"demo_0": {"wrist": _video(20, 10)}})

    results = cli.annotate(path, task="Scan")

    with pytest.raises(ValueError, match="no successful episodes"):
        cli.filter_successful(path, tmp_path / "filtered.hdf5", results)


def test_mimic_variants_share_their_source_verdict_without_a_second_call(tmp_path, vlm) -> None:
    source, other = _video(20, 200), _video(20, 10)
    path = _recording(
        tmp_path / "r.hdf5",
        {
            "demo_0": {"wrist": source},
            "demo_1": {"wrist": other},
            "demo_2": {"wrist": source.copy()},  # a mimic clone of demo_0
            "demo_3": {"wrist": other.copy()},  # and of demo_1
        },
    )

    results = cli.annotate(path, task="Scan", write=True)

    assert len(vlm.calls) == 2
    assert [success for _, success, _ in results] == [True, False, True, False]
    with h5py.File(path, "r") as handle:
        assert handle["data/demo_2"].attrs["vlm_success_from"] == "demo_0"
        assert handle["data/demo_3"].attrs["vlm_success_from"] == "demo_1"
        assert "vlm_success_from" not in handle["data/demo_0"].attrs


def test_an_unknown_camera_is_an_error(tmp_path, vlm) -> None:
    path = _recording(tmp_path / "r.hdf5", {"demo_0": {"room": _video(5, 200)}})

    with pytest.raises(KeyError, match="no camera 'wrist'"):
        cli.annotate(path, task="Scan", camera=["room", "wrist"])


def test_dry_run_reports_what_would_be_sent(tmp_path, vlm) -> None:
    path = _recording(tmp_path / "r.hdf5", {"demo_0": {"room": _video(30, 200), "wrist": _video(30, 200)}})

    results = cli.annotate(path, task="Scan", camera="room,wrist", judge="trajectory", dry_run=True)

    assert results == [("demo_0", False, "dry-run: 6 frames from room, 6 frames from wrist (trajectory)")]
    assert vlm.calls == []


def test_main_accepts_repeated_cameras_and_a_judge(tmp_path, vlm, capsys) -> None:
    path = _recording(tmp_path / "r.hdf5", {"demo_0": {"room": _video(12, 200), "wrist": _video(12, 200)}})

    code = cli.main(
        ["--task", "Scan", "--camera", "room", "--camera", "wrist", "--judge", "trajectory", "offline", str(path)]
    )

    assert code == 0
    assert "1/1 succeeded" in capsys.readouterr().out
    (content,) = vlm.calls
    assert [item["text"] for item in content if item["type"] == "text"][1:] == ["Camera room:", "Camera wrist:"]


def test_live_grading_takes_one_camera(capsys) -> None:
    assert cli.main(["--task", "Scan", "--camera", "room,wrist", "live", "--namespace", "x"]) == 2
    assert "one camera" in capsys.readouterr().out

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The glue between the manifest, the pinned checkouts and the model, without the model.

Everything here is the pure part of what ``server.py`` and ``train.py`` do
right before they touch gr00t or torch, so it runs in the light venv.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.server import (  # noqa: E402
    ServeOptions,
    build_tactile_provider,
    checkpoint_config,
    checkpoint_fingertips,
    refresh_fields,
    served_info,
    urdf_overrides,
)
from i4h_tasks.gr00t_n17_tactile.tactile import sharpa  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.train import launcher_command, resolved_argv  # noqa: E402

ROOT = Path(__file__).resolve().parents[3]


# -- the public checkouts ---------------------------------------------------


def test_the_encoder_checkouts_live_under_third_party() -> None:
    assert sharpa.public_source_paths() == [
        ROOT / "third_party" / "sharpawave-deform-encoder",
        ROOT / "third_party" / "T-Rex",
    ]


def test_a_missing_checkout_names_the_setup_command(monkeypatch, tmp_path) -> None:
    monkeypatch.setattr(sharpa, "public_source_paths", lambda: [tmp_path / "absent"])
    with pytest.raises(FileNotFoundError, match="setup.sh tasks/gr00t_n17_tactile"):
        sharpa.import_public_sources()


def test_present_checkouts_go_on_sys_path_once(monkeypatch, tmp_path) -> None:
    checkout = tmp_path / "pkg"
    checkout.mkdir()
    monkeypatch.setattr(sharpa, "public_source_paths", lambda: [checkout])
    monkeypatch.setattr(sys, "path", list(sys.path))
    sharpa.import_public_sources()
    sharpa.import_public_sources()
    assert sys.path.count(str(checkout)) == 1


def test_the_urdf_is_the_base_variant_inside_its_directory() -> None:
    assert sharpa.urdf_file("wave_01/left_sharpa_wave", "left") == Path(
        "wave_01/left_sharpa_wave/left_sharpa_wave.urdf"
    )
    with pytest.raises(ValueError, match="hand"):
        sharpa.urdf_file("wave_01/left_sharpa_wave", "upper")


# -- what the trainer hands the launcher ----------------------------------


def test_launcher_command_uses_torchrun_only_for_several_gpus() -> None:
    argv = ["gr00t/experiment/launch_finetune.py", "--max_steps", "20", "--num_gpus", "2"]
    assert launcher_command(argv, python="py") == [
        "py",
        "-m",
        "torch.distributed.run",
        "--nproc_per_node",
        "2",
        "--master_port",
        "29500",
        *argv,
    ]
    assert launcher_command(argv[:3], python="py") == ["py", *argv[:3]]
    assert launcher_command(argv[:3] + ["--num_gpus", "1"], python="py")[1] == argv[0]


def test_resolved_argv_substitutes_only_the_mapped_uris(tmp_path) -> None:
    argv = ["launch.py", "--sharpa_encoder_ckpt_path", "hf://repo/file.safetensors", "--max_steps", "5"]
    out = resolved_argv(argv, {"hf://repo/file.safetensors": tmp_path / "file.safetensors"})
    assert out == ["launch.py", "--sharpa_encoder_ckpt_path", str(tmp_path / "file.safetensors"), "--max_steps", "5"]
    assert resolved_argv(argv, {}) == argv


# -- what the server reads from a checkpoint and feeds the model ------------


def test_checkpoint_config_reads_the_saved_json(tmp_path) -> None:
    (tmp_path / "config.json").write_text(json.dumps({"use_tactile": True, "sharpa_right_urdf_path": None}))
    assert checkpoint_config(str(tmp_path)) == {"use_tactile": True, "sharpa_right_urdf_path": None}
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError, match="config.json"):
        checkpoint_config(str(tmp_path / "empty"))
    with pytest.raises(FileNotFoundError, match="checkpoint directory"):
        checkpoint_config(str(tmp_path / "missing"))


def test_refresh_fields_keep_maps_and_the_used_hands_only() -> None:
    frame = {
        "deform_maps": np.zeros((1, 10, 1, 240, 240), dtype=np.uint8),
        "timestamp": np.float64(1.0),
        "tactile_force": np.zeros((1, 10, 16, 6), dtype=np.float32),
        "joint_angles_left": np.zeros((1, 22), dtype=np.float32),
        "joint_angles_right": np.zeros((1, 22), dtype=np.float32),
    }
    assert set(refresh_fields(frame)) == {"deform_maps", "joint_angles_left", "joint_angles_right"}
    assert set(refresh_fields(frame, hands=("left",))) == {"deform_maps", "joint_angles_left"}


def test_hf_uris_round_trip_with_and_without_a_revision() -> None:
    pinned = sharpa.hf_uri("org/repo", "weights.safetensors", "abc123")
    assert pinned == "hf://org/repo@abc123/weights.safetensors"
    assert sharpa.parse_hf_uri(pinned) == ("org/repo", "abc123", "weights.safetensors")
    assert sharpa.parse_hf_uri(sharpa.hf_uri("org/repo", "model.pt")) == ("org/repo", None, "model.pt")
    with pytest.raises(ValueError):
        sharpa.parse_hf_uri("s3://bucket/file")


# -- which URDFs a checkpoint gets -----------------------------------------

ASSETS = {"left_urdf": Path("/pin/left.urdf"), "right_urdf": Path("/pin/right.urdf")}


def test_a_bimanual_checkpoint_gets_both_urdfs_overridden() -> None:
    saved = {"sharpa_left_urdf_path": "/workspace/left.urdf", "sharpa_right_urdf_path": "/workspace/right.urdf"}
    assert urdf_overrides(saved, ASSETS) == {
        "sharpa_left_urdf_path": "/pin/left.urdf",
        "sharpa_right_urdf_path": "/pin/right.urdf",
    }


def test_a_single_hand_checkpoint_keeps_its_missing_hand_unset() -> None:
    saved = {"sharpa_left_urdf_path": "/workspace/left.urdf", "sharpa_right_urdf_path": None}
    assert urdf_overrides(saved, ASSETS) == {"sharpa_left_urdf_path": "/pin/left.urdf"}
    assert urdf_overrides({}, ASSETS) == {}


# -- the checkpoint's fingertips --------------------------------------------


def test_checkpoint_fingertips_follow_the_tactile_keys() -> None:
    left = [f"{name}_deform" for name in FINGERTIPS[:5]]
    assert checkpoint_fingertips(left) == list(FINGERTIPS[:5])
    assert checkpoint_fingertips(reversed(left)) == list(reversed(FINGERTIPS[:5]))
    with pytest.raises(ValueError, match="left_sixth_deform"):
        checkpoint_fingertips(["left_sixth_deform"])


# -- what the provider hands the model --------------------------------------


class _FakeSource:
    """A live source that is already complete, fed by hand."""

    last = None

    def __init__(self, host, port, force_window=16, fingertips=FINGERTIPS):
        self.fingertips = tuple(fingertips)
        self.frame = None
        _FakeSource.last = self

    def wait_ready(self, timeout):
        pass

    def missing(self):
        return []

    def latest(self):
        return self.frame


def test_the_provider_returns_maps_and_joints_only(monkeypatch) -> None:
    from i4h_tasks.gr00t_n17_tactile.tactile.interface import TactileFrame

    monkeypatch.setattr(sharpa, "SharpaLiveSource", _FakeSource)
    provider = build_tactile_provider(None, ServeOptions(), FINGERTIPS[:5])
    assert _FakeSource.last.fingertips == FINGERTIPS[:5], "the source is opened for the checkpoint's fingertips"
    assert provider() is None, "no frame yet keeps the previous step's inputs"
    _FakeSource.last.frame = TactileFrame(
        timestamp=1.0,
        deform=np.zeros((5, 240, 240), dtype=np.uint8),
        force=np.zeros((5, 6), dtype=np.float32),
        hand_joints={"left": np.zeros(22, dtype=np.float32)},
        fingertips=FINGERTIPS[:5],
    )
    inputs = provider()
    assert set(inputs) == {"deform_maps", "joint_angles_left"}
    assert inputs["deform_maps"].shape == (1, 5, 1, 240, 240)
    assert inputs["joint_angles_left"].shape == (1, 22)


def test_the_provider_is_off_without_a_live_source() -> None:
    assert build_tactile_provider(None, ServeOptions(tactile_source="none")) is None
    assert build_tactile_provider(None, ServeOptions(disable_tactile=True)) is None


# -- what the client is told about the loaded checkpoint --------------------


def _policy(tactile_keys, video_keys, state_keys, **config):
    configs = {
        "video": SimpleNamespace(modality_keys=video_keys),
        "state": SimpleNamespace(modality_keys=state_keys),
        "language": SimpleNamespace(modality_keys=["annotation.human.task_description"]),
        "tactile": SimpleNamespace(modality_keys=tactile_keys),
        "tactile_force": SimpleNamespace(modality_keys=["force"]),
    }
    model = type("Gr00tN1d7Tactile", (), {})()
    model.config = SimpleNamespace(**config)
    return SimpleNamespace(modality_configs=configs, model=model)


def test_served_info_describes_the_loaded_checkpoint_not_the_manifest() -> None:
    manifest_info = {
        "video_modality_keys": ["high", "left_wrist_view", "right_wrist_view"],
        "state_groups": {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 36], "right_hand": [36, 58]},
        "tactile": {"enabled": True, "fingertips": list(FINGERTIPS), "force_window": 16, "delay_aug": True},
    }
    policy = _policy(
        tactile_keys=[f"{name}_deform" for name in FINGERTIPS[:5]],
        video_keys=["high", "right_wrist_view"],
        state_keys=["left_arm", "right_arm", "left_hand"],
        use_tactile=True,
        use_tactile_force=False,
        delay_aug=False,
        force_delay_max=0,
        num_tactile_tokens=5,
    )
    info = served_info(manifest_info, policy, ServeOptions(disable_tactile=True))
    assert info["video_modality_keys"] == ["high", "right_wrist_view"]
    assert list(info["state_groups"]) == ["left_arm", "right_arm", "left_hand"]
    assert info["tactile"]["fingertips"] == list(FINGERTIPS[:5]) and info["tactile"]["num_tokens"] == 5
    assert info["tactile"]["force_window"] == 0 and info["tactile"]["delay_aug"] is False
    assert info["checkpoint"]["class"] == "Gr00tN1d7Tactile" and info["checkpoint"]["num_tactile_tokens"] == 5
    assert info["required_observation_keys"]["tactile"] == [f"{name}_deform" for name in FINGERTIPS[:5]]
    assert info["required_observation_keys"]["tactile_force"] == ["force"]
    assert any("tactile_force keys" in note for note in info["notes"])
    assert manifest_info["tactile"]["fingertips"] == list(FINGERTIPS), "the manifest view is untouched"

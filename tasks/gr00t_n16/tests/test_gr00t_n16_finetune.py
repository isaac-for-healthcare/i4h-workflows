# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n16 import _finetune
from i4h_tasks.gr00t_n16.train import main


def test_run_pins_requested_gpus_before_cuda_query(monkeypatch) -> None:
    monkeypatch.delenv("CUDA_VISIBLE_DEVICES", raising=False)

    def available_gpus() -> int:
        assert _finetune.os.environ["CUDA_VISIBLE_DEVICES"] == "0"
        return 1

    monkeypatch.setattr(_finetune, "_available_gpus", available_gpus)
    monkeypatch.setattr(_finetune, "_groot_root", lambda: _finetune.Path("/missing"))

    with pytest.raises(SystemExit, match="source not found"):
        _finetune.run(_finetune.TrainConfig(dataset_path=["dataset"], num_gpus=1))


def test_run_preserves_explicit_gpu_selection(monkeypatch) -> None:
    monkeypatch.setenv("CUDA_VISIBLE_DEVICES", "1")
    monkeypatch.setattr(_finetune, "_available_gpus", lambda: 1)
    monkeypatch.setattr(_finetune, "_groot_root", lambda: _finetune.Path("/missing"))

    with pytest.raises(SystemExit, match="source not found"):
        _finetune.run(_finetune.TrainConfig(dataset_path=["dataset"], num_gpus=1))

    assert _finetune.os.environ["CUDA_VISIBLE_DEVICES"] == "1"


def test_train_uses_task_modality_config(monkeypatch, tmp_path, capsys) -> None:
    monkeypatch.setattr("i4h_common.training.resolve_dataset", lambda path: path)
    monkeypatch.setattr(
        "i4h_common.training.require_trainable",
        lambda _task_id: SimpleNamespace(train={"modality_config_path": "custom_headcam.py"}),
    )

    checked = []
    monkeypatch.setattr(_finetune, "check_datasets", checked.append)

    assert main(["--task", "gr00t_n16/test_task", "--dataset", str(tmp_path), "--dry-run"]) == 0

    assert "custom_headcam.py" in capsys.readouterr().out
    assert checked[0].modality_config_path == "custom_headcam.py"


def _converted(tmp_path: Path, *, recorded: list[str], modality_video: dict) -> Path:
    meta = tmp_path / "meta"
    meta.mkdir()
    features = {"observation.state": {}, "action": {}}
    features.update({f"observation.images.{camera}": {"dtype": "video"} for camera in recorded})
    (meta / "info.json").write_text(json.dumps({"features": features}))
    (meta / "modality.json").write_text(json.dumps({"video": modality_video}))
    return tmp_path


def test_default_modality_config_is_the_head_camera() -> None:
    """The G1 scenes record ``head`` alone and the Rheo checkpoints were trained on ``ego_view``."""
    assert _finetune._modality_config_path(_finetune.TrainConfig(dataset_path=["d"])).name == "config_head.py"
    assert _finetune.DEFAULT_MODALITY_CONFIG.is_file()


def _registered_video_keys(monkeypatch, config_name: str) -> list[str]:
    """Load a shipped config against stub gr00t modules and return its video keys."""
    registered = {}

    class _ModalityConfig:
        def __init__(self, **kwargs) -> None:
            self.__dict__.update(kwargs)

    enum = SimpleNamespace(NEW_EMBODIMENT=SimpleNamespace(value="new_embodiment"))
    stubs = {
        "gr00t": ModuleType("gr00t"),
        "gr00t.configs": ModuleType("gr00t.configs"),
        "gr00t.configs.data": ModuleType("gr00t.configs.data"),
        "gr00t.configs.data.embodiment_configs": SimpleNamespace(
            register_modality_config=lambda config, embodiment_tag: registered.update(config)
        ),
        "gr00t.data": ModuleType("gr00t.data"),
        "gr00t.data.embodiment_tags": SimpleNamespace(EmbodimentTag=enum),
        "gr00t.data.types": SimpleNamespace(
            ActionConfig=lambda **kwargs: kwargs,
            ActionFormat=SimpleNamespace(DEFAULT="default"),
            ActionRepresentation=SimpleNamespace(ABSOLUTE="absolute"),
            ActionType=SimpleNamespace(NON_EEF="non_eef"),
            ModalityConfig=_ModalityConfig,
        ),
    }
    for name, module in stubs.items():
        monkeypatch.setitem(sys.modules, name, module)
    monkeypatch.delitem(sys.modules, "i4h_tasks.gr00t_n16.locomanip.modality", raising=False)
    _finetune._load_modality_config(_finetune.DEFAULT_MODALITY_CONFIG.parent / config_name)
    return registered["video"].modality_keys


def test_head_config_requests_only_the_ego_view(monkeypatch) -> None:
    assert _registered_video_keys(monkeypatch, "config_head.py") == ["ego_view"]


def test_dualcam_config_keeps_both_views(monkeypatch) -> None:
    assert _registered_video_keys(monkeypatch, "config_dualcam.py") == ["ego_view", "room"]


def test_head_only_recording_passes_the_head_config(tmp_path) -> None:
    dataset = _converted(
        tmp_path, recorded=["head"], modality_video={"ego_view": {"original_key": "observation.images.head"}}
    )

    _finetune.check_dataset_videos(str(dataset), ["ego_view"], _finetune.DEFAULT_MODALITY_CONFIG)


def test_a_config_asking_for_an_unrecorded_camera_fails_before_training(tmp_path) -> None:
    """NVBug 6925769: the dual-camera config failed at the first batch on head-only G1 data."""
    dataset = _converted(
        tmp_path,
        recorded=["head"],
        modality_video={
            "ego_view": {"original_key": "observation.images.head"},
            "room": {"original_key": "observation.images.room"},
        },
    )

    with pytest.raises(
        SystemExit, match=r"'room' \(observation.images.room, not in meta/info.json\).*\['observation.images.head'\]"
    ):
        _finetune.check_dataset_videos(str(dataset), ["ego_view", "room"], Path("config_dualcam.py"))


def test_a_config_key_missing_from_modality_json_fails(tmp_path) -> None:
    dataset = _converted(
        tmp_path, recorded=["head"], modality_video={"ego_view": {"original_key": "observation.images.head"}}
    )

    with pytest.raises(SystemExit, match=r"'room' \(no entry in meta/modality.json\)"):
        _finetune.check_dataset_videos(str(dataset), ["ego_view", "room"], Path("config_dualcam.py"))


def test_a_hub_dataset_is_not_checked_locally() -> None:
    _finetune.check_dataset_videos("nvidia/some-dataset", ["ego_view", "room"], Path("config_dualcam.py"))


G1_TASKS = ["gr00t_n16/locomanip_push_cart", "gr00t_n16/locomanip_tray_pick_and_place"]


@pytest.mark.parametrize("task_id", G1_TASKS)
def test_the_g1_manifests_declare_the_head_camera_config(task_id) -> None:
    """The camera contract is a training default, so it lives in the task manifest (AGENTS.md)."""
    from i4h_common.training import require_trainable

    declared = require_trainable(task_id).train["modality_config_path"]

    assert _finetune.resolve_modality_config_path(declared) == _finetune.DEFAULT_MODALITY_CONFIG


@pytest.mark.parametrize("task_id", G1_TASKS)
def test_trainer_and_server_load_the_same_declared_config(monkeypatch, tmp_path, task_id) -> None:
    from i4h_common.server import Session
    from i4h_tasks.gr00t_n16.server import Gr00tN16Server

    monkeypatch.setattr("i4h_common.training.resolve_dataset", lambda path: path)
    trained = []
    monkeypatch.setattr(_finetune, "check_datasets", trained.append)
    assert main(["--task", task_id, "--dataset", str(tmp_path), "--dry-run"]) == 0

    served = []
    monkeypatch.setattr(_finetune, "_load_modality_config", served.append)
    monkeypatch.setitem(
        sys.modules,
        "i4h_tasks.gr00t_n16.locomanip.infer.closedloop_policy",
        SimpleNamespace(G1LocomanipClosedloopPolicy=lambda **_kwargs: SimpleNamespace(reset=lambda: None)),
    )
    server = object.__new__(Gr00tN16Server)
    server._policies = {}
    server.manifest_dir = None
    server.load(
        Session(task_uid="t", task_id=task_id, run_id="r", episode_index=0, prompt="p", checkpoint="c", model={})
    )

    resolved = _finetune.resolve_modality_config_path(trained[0].modality_config_path)
    assert resolved == served[0] == _finetune.DEFAULT_MODALITY_CONFIG


def test_a_local_dataset_without_groot_metadata_fails_the_check(tmp_path) -> None:
    """Greptile on PR 275: a dataset with info.json but no modality.json used to pass unchecked."""
    meta = tmp_path / "meta"
    meta.mkdir()
    (meta / "info.json").write_text(json.dumps({"features": {"observation.images.head": {}}}))

    with pytest.raises(SystemExit, match=r"not a GR00T LeRobot dataset; it has no meta/modality.json"):
        _finetune.check_dataset_videos(str(tmp_path), ["ego_view"], _finetune.DEFAULT_MODALITY_CONFIG)
    with pytest.raises(SystemExit, match=r"meta/info.json or meta/modality.json"):
        _finetune.check_dataset_videos(str(tmp_path / "meta"), ["ego_view"], _finetune.DEFAULT_MODALITY_CONFIG)

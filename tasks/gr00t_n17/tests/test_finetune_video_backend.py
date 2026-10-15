# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pick a camera-video decoder the fine-tune can actually import.

GR00T defaults to torchcodec and raises instead of falling back when it is
missing, which fails the first DataLoader batch. A ``--dry-run`` never decodes a
frame, so it passed on Linux aarch64 while every real run there died at step 0.
"""

from __future__ import annotations

import dataclasses
import importlib
import types
from pathlib import Path

import pytest

from i4h_tasks.gr00t_n17 import _finetune

TASK_ID = "gr00t_n17/catheter_navigation"


def _only_importable(monkeypatch, available: set[str], failure: Exception | None = None) -> None:
    real_import = importlib.import_module

    def import_module(name: str, *args, **kwargs):
        if name not in _finetune.VIDEO_BACKENDS:
            return real_import(name, *args, **kwargs)
        if name not in available:
            raise failure or ModuleNotFoundError(f"No module named {name!r}", name=name)
        return types.ModuleType(name)

    monkeypatch.setattr(_finetune.importlib, "import_module", import_module)


def test_torchcodec_wins_where_it_imports(monkeypatch) -> None:
    _only_importable(monkeypatch, {"torchcodec", "decord"})
    assert _finetune.resolve_video_backend() == "torchcodec"


def test_decord_is_used_where_torchcodec_is_missing(monkeypatch, caplog) -> None:
    """The Linux aarch64 install: decord2 present, no torchcodec."""
    _only_importable(monkeypatch, {"decord"})
    with caplog.at_level("INFO", logger=_finetune.logger.name):
        assert _finetune.TrainConfig(dataset_path=["d"]).video_backend == "decord"
    assert "torchcodec is not installed" in caplog.text


@pytest.mark.parametrize(
    "failure",
    [
        RuntimeError("Could not load libtorchcodec"),
        OSError("libavutil.so.58: cannot open shared object file"),
        ModuleNotFoundError("No module named 'torch._C'", name="torch._C"),
    ],
)
def test_an_installed_torchcodec_that_fails_to_import_stops_the_run(monkeypatch, failure) -> None:
    """A broken install must not quietly switch decoder; the error names why."""
    _only_importable(monkeypatch, {"decord"}, failure=failure)
    with pytest.raises(SystemExit, match=r"torchcodec is installed but failed to import") as caught:
        _finetune.resolve_video_backend()
    assert str(failure) in str(caught.value)
    assert "--video-backend decord" in str(caught.value)
    assert caught.value.__cause__ is failure


def test_no_backend_fails_before_training_starts(monkeypatch) -> None:
    _only_importable(monkeypatch, set())
    with pytest.raises(SystemExit, match="no video backend"):
        _finetune.TrainConfig(dataset_path=["d"])


def test_an_explicit_backend_is_kept(monkeypatch) -> None:
    _only_importable(monkeypatch, {"torchcodec", "decord"})
    assert _finetune.TrainConfig(dataset_path=["d"], video_backend="decord").video_backend == "decord"


def test_the_dry_run_reports_the_backend_a_real_run_would_use(tmp_path: Path, capsys, monkeypatch) -> None:
    from i4h_tasks.gr00t_n17.train import main

    _only_importable(monkeypatch, {"decord"})
    assert main(["--task", TASK_ID, "--dataset", str(tmp_path), "--dry-run"]) == 0
    assert "video_backend = 'decord'" in capsys.readouterr().out


def test_the_cli_flag_overrides_the_automatic_choice(tmp_path: Path, capsys) -> None:
    from i4h_tasks.gr00t_n17.train import main

    assert main(["--task", TASK_ID, "--dataset", str(tmp_path), "--video-backend", "decord", "--dry-run"]) == 0
    assert "video_backend = 'decord'" in capsys.readouterr().out


def _dataset_with_video(root: Path, encoder: str) -> Path:
    """Write a five-frame camera video where LeRobot v2.1 keeps them."""
    av = pytest.importorskip("av")
    import numpy as np

    try:
        av.codec.Codec(encoder, "w")
    except Exception:  # noqa: BLE001 - PyAV raises UnknownCodecError or a plain ValueError by version
        pytest.skip(f"PyAV has no {encoder} encoder")
    video = root / "videos" / "chunk-000" / "observation.images.cam" / "episode_000000.mp4"
    video.parent.mkdir(parents=True)
    with av.open(str(video), "w") as container:
        stream = container.add_stream(encoder, rate=30)
        stream.width, stream.height, stream.pix_fmt = 64, 64, "yuv420p"
        for _ in range(5):
            frame = av.VideoFrame.from_ndarray(np.zeros((64, 64, 3), np.uint8), format="rgb24")
            container.mux(stream.encode(frame))
        container.mux(stream.encode())
    return root


def test_an_av1_dataset_is_refused_before_decord_trains_on_it(tmp_path: Path) -> None:
    dataset = _dataset_with_video(tmp_path, "libsvtav1")
    with pytest.raises(SystemExit, match=r"encoded as av1.*--video-codec h264"):
        _finetune.check_video_codecs(str(dataset), "decord")


def test_an_h264_dataset_trains_with_decord(tmp_path: Path) -> None:
    _finetune.check_video_codecs(str(_dataset_with_video(tmp_path, "libx264")), "decord")


def test_torchcodec_reads_av1_so_its_datasets_are_not_probed(tmp_path: Path) -> None:
    _finetune.check_video_codecs(str(_dataset_with_video(tmp_path, "libsvtav1")), "torchcodec")


def test_a_dataset_without_videos_is_not_probed(tmp_path: Path) -> None:
    _finetune.check_video_codecs(str(tmp_path), "decord")


def test_the_dry_run_refuses_an_av1_dataset_for_decord(tmp_path: Path) -> None:
    from i4h_tasks.gr00t_n17.train import main

    dataset = _dataset_with_video(tmp_path, "libsvtav1")
    with pytest.raises(SystemExit, match="encoded as av1"):
        main(["--task", TASK_ID, "--dataset", str(dataset), "--video-backend", "decord", "--dry-run"])


def test_the_backend_reaches_the_shim_ahead_of_the_launch_script() -> None:
    script = Path("/groot/gr00t/experiment/launch_finetune.py")
    cmd = _finetune._build_cmd(_finetune.TrainConfig(dataset_path=["d"], video_backend="decord"), script, "d")
    assert cmd.index("--video-backend=decord") < cmd.index(str(script))


def test_the_shim_files_the_backend_under_the_data_config() -> None:
    from i4h_tasks.gr00t_n17._launcher import split_shim_args

    overrides, rest = split_shim_args(["--video-backend=decord", "launch.py", "--max_steps", "5"])
    assert overrides == {"data": {"video_backend": "decord"}}
    assert rest == ["launch.py", "--max_steps", "5"]


def test_without_options_the_shim_passes_the_argv_through() -> None:
    from i4h_tasks.gr00t_n17._launcher import split_shim_args

    assert split_shim_args(["launch.py", "--max_steps", "5"]) == ({}, ["launch.py", "--max_steps", "5"])


def test_the_shim_rejects_an_unknown_backend() -> None:
    from i4h_tasks.gr00t_n17._launcher import split_shim_args

    with pytest.raises(SystemExit):
        split_shim_args(["--video-backend=opencv", "launch.py"])


def test_the_backend_survives_the_launchers_own_config_load(monkeypatch) -> None:
    """``launch_finetune.py`` calls ``get_default_config().load_dict(...)`` and the loader reads ``config.data``."""
    base_config = pytest.importorskip("gr00t.configs.base_config", reason="needs the gr00t_n17 venv")
    from i4h_tasks.gr00t_n17._launcher import install_config_overrides

    assert base_config.get_default_config().data.video_backend == "torchcodec"
    monkeypatch.setattr(base_config, "get_default_config", base_config.get_default_config)
    monkeypatch.setattr(base_config.Config, "load_dict", base_config.Config.load_dict)
    install_config_overrides({"data": {"video_backend": "decord"}})

    config = base_config.get_default_config().load_dict({"data": {"download_cache": False, "datasets": []}})
    assert config.data.video_backend == "decord"


def test_the_saved_config_still_loads(monkeypatch, tmp_path: Path) -> None:
    """The checkpoint's ``config.yaml`` is ``yaml.dump(config)`` and is read back with ``yaml.Loader``."""
    base_config = pytest.importorskip("gr00t.configs.base_config", reason="needs the gr00t_n17 venv")
    yaml = pytest.importorskip("yaml")
    from i4h_tasks.gr00t_n17._launcher import install_config_overrides

    monkeypatch.setattr(base_config, "get_default_config", base_config.get_default_config)
    monkeypatch.setattr(base_config.Config, "load_dict", base_config.Config.load_dict)
    install_config_overrides({"data": {"video_backend": "decord"}})
    config = base_config.get_default_config().load_dict({"data": {"download_cache": False, "datasets": []}})

    assert set(vars(config)) == {f.name for f in dataclasses.fields(base_config.Config)}
    restored = yaml.load(yaml.dump(config), Loader=yaml.Loader)
    assert restored.data.video_backend == "decord"

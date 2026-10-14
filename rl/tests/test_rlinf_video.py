# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise pinned RLinf encoder methods with actual CPU codecs."""

import ast
import os
import sys
import warnings
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from types import MethodType, SimpleNamespace

import numpy as np
import pytest

from i4h_rl.backends import rlinf

imageio = pytest.importorskip("imageio")
pytest.importorskip("av")
iio = pytest.importorskip("imageio.v3")

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recorder(tmp_path):
    # Import the real methods without requiring GR00T/Ray/Isaac Sim to import
    # the wrapper module. The source is the pinned checkout patched by setup.
    path = ROOT / "third_party/RLinf-a4b6abe/rlinf/envs/wrappers/record_video.py"
    if not path.is_file():
        pytest.skip(f"optional patched RLinf source is absent: {path}")
    tree = ast.parse(path.read_text())
    methods = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RecordVideo")
    context = SimpleNamespace(
        _fps=30,
        raise_on_error=False,
        _executor=ThreadPoolExecutor(1),
        _save_futures=[],
        render_images=[],
        video_cnt=0,
        env=SimpleNamespace(seed=0),
        video_cfg=SimpleNamespace(video_base_dir=str(tmp_path)),
    )
    namespace = dict(imageio=imageio, np=np, os=os, warnings=warnings, Future=Future, Optional=object)
    for method in methods.body:
        if isinstance(method, ast.FunctionDef) and method.name in (
            "flush_video",
            "_submit_save",
            "_save_video",
            "_prune_futures",
        ):
            # Type annotations are not needed to execute the encoder methods.
            method.returns = None
            for arg in method.args.args:
                arg.annotation = None
            exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"), namespace)
            setattr(context, method.name, MethodType(namespace[method.name], context))
    yield context
    context._executor.shutdown(wait=True)


def test_video_flush_writes_decodable_mp4(recorder, tmp_path):
    recorder.render_images = [np.full((64, 64, 3), value, np.uint8) for value in (20, 80, 160)]
    recorder.flush_video()
    path = tmp_path / "seed_0/0.mp4"
    frames = iio.imread(path, plugin="pyav")
    assert len(frames) == 3 and frames[0].shape == (64, 64, 3)
    assert abs(float(frames[-1].mean()) - 160) < 3


@pytest.mark.parametrize("strict", [False, True])
@pytest.mark.parametrize("stage", ["open", "append", "close"])
def test_encoder_failure_policy(recorder, monkeypatch, strict, stage):
    recorder.raise_on_error = strict
    closed = []

    class BrokenWriter:
        def append_data(self, _frame):
            if stage == "append":
                raise OSError("encoder failed")

        def close(self):
            closed.append(True)
            if stage == "close":
                raise OSError("encoder failed")

    def get_writer(*args, **kwargs):
        if stage == "open":
            raise OSError("encoder failed")
        return BrokenWriter()

    monkeypatch.setattr(imageio, "get_writer", get_writer)
    recorder.render_images = [np.zeros((64, 64, 3), np.uint8)]
    expectation = (
        pytest.raises(RuntimeError, match="encoder failed")
        if strict
        else pytest.warns(UserWarning, match="encoder failed")
    )
    with expectation:
        recorder.flush_video()
    assert bool(closed) == (stage != "open")
    assert recorder._save_futures[0].done()


def test_evaluation_rejects_corrupt_or_empty_video(tmp_path):
    directory = tmp_path / "video/eval/seed_0"
    directory.mkdir(parents=True)
    path = directory / "0.mp4"
    assert not rlinf.verify_evaluation_video(tmp_path / "video/eval", Path(sys.executable), os.environ.copy())
    path.touch()
    assert not rlinf.verify_evaluation_video(tmp_path / "video/eval", Path(sys.executable), os.environ.copy())
    path.write_bytes(b"not an mp4")
    assert not rlinf.verify_evaluation_video(tmp_path / "video/eval", Path(sys.executable), os.environ.copy())


def test_evaluation_accepts_decodable_video(recorder, tmp_path):
    directory = tmp_path / "video/eval/seed_0"
    directory.mkdir(parents=True)
    recorder._save_video([np.zeros((64, 64, 3), np.uint8)], str(directory / "0.mp4"))
    assert rlinf.verify_evaluation_video(tmp_path / "video/eval", Path(sys.executable), os.environ.copy())


@pytest.mark.parametrize("custom_dir", [False, True])
@pytest.mark.parametrize("requested_by", ["flag", "config"])
@pytest.mark.parametrize("contents", [None, b"corrupt mp4"])
def test_launcher_reports_video_failure_without_metrics_error(
    tmp_path, monkeypatch, capsys, requested_by, contents, custom_dir
):
    import argparse

    from i4h_rl.profile import RLProfile

    profile = RLProfile.load(ROOT / "rl/profiles/assemble_trocar.yaml")
    run_dir = tmp_path / "run"
    video_dir = tmp_path / "custom_videos" if custom_dir else run_dir / "video/eval"
    args = argparse.Namespace(
        resolved_model_path=tmp_path,
        only_eval=True,
        rl_model_path="checkpoint",
        runtime_python=None,
        sim_runtime_python=None,
        resolved_model_gpu="0",
        resolved_sim_gpu="1",
        run_dir=str(run_dir),
        video=requested_by == "flag",
        resume_dir=None,
    )
    monkeypatch.setattr(rlinf, "weights", lambda _: tmp_path)
    monkeypatch.setattr(rlinf, "checkpoint_root", lambda _: tmp_path)
    monkeypatch.setattr(rlinf, "_model_runtime_python", lambda *_: Path(sys.executable))
    monkeypatch.setattr(rlinf, "_sim_runtime_python", lambda *_: Path(sys.executable))
    monkeypatch.setattr(rlinf, "_model_preflight", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(rlinf, "_sim_preflight", lambda *_: None)
    monkeypatch.setattr(rlinf, "_sim_env", lambda *_args, **_kwargs: {})
    monkeypatch.setattr(rlinf, "_runtime_env", lambda *_: os.environ.copy())
    monkeypatch.setattr(
        rlinf, "finalize_evaluation", lambda *_args, **_kwargs: pytest.fail("metrics must not mask video failure")
    )

    class Simulator:
        def __init__(self, *args, **kwargs):
            (run_dir / "simulator.ready").touch()
            if contents is not None:
                video_dir.mkdir(parents=True)
                (video_dir / "0.mp4").write_bytes(contents)

        def poll(self):
            return 0

    original_popen = rlinf.subprocess.Popen
    monkeypatch.setattr(rlinf.subprocess, "Popen", Simulator)
    monkeypatch.setattr(rlinf.subprocess, "call", lambda *_args, **_kwargs: 0)
    # Video decoding uses subprocess.run; keep its real Popen implementation.
    real_run = rlinf.subprocess.run

    def probe(*args, **kwargs):
        with monkeypatch.context() as context:
            context.setattr(rlinf.subprocess, "Popen", original_popen)
            return real_run(*args, **kwargs)

    monkeypatch.setattr(rlinf.subprocess, "run", probe)
    overrides = ("env.eval.video_cfg.save_video=true",) if requested_by == "config" else ()
    if custom_dir:
        overrides += (f"env.eval.video_cfg.video_base_dir={video_dir}",)
    assert rlinf.launch(args, profile, ROOT, num_envs=1, epochs=1, overrides=overrides) == 1
    error = capsys.readouterr().err
    assert "requested evaluation video" in error
    if contents is None:
        assert str(video_dir) in error
    assert "metrics" not in error


@pytest.mark.parametrize("configured", [None, False, True])
def test_wrapper_reads_strict_option_with_optional_default(configured):
    path = ROOT / "third_party/RLinf-a4b6abe/rlinf/envs/wrappers/record_video.py"
    if not path.is_file():
        pytest.skip(f"optional patched RLinf source is absent: {path}")
    tree = ast.parse(path.read_text())
    wrapper = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RecordVideo")
    initializer = next(node for node in wrapper.body if isinstance(node, ast.FunctionDef) and node.name == "__init__")
    initializer.returns = None
    for argument in initializer.args.args:
        argument.annotation = None
    namespace = dict(gym=SimpleNamespace(Env=type("Env", (), {})), ThreadPoolExecutor=ThreadPoolExecutor)
    exec(compile(ast.Module(body=[initializer], type_ignores=[]), str(path), "exec"), namespace)
    context = SimpleNamespace()
    config = SimpleNamespace() if configured is None else SimpleNamespace(raise_on_error=configured)
    namespace["__init__"](context, SimpleNamespace(seed=0), config, fps=30)
    try:
        assert context.raise_on_error == bool(configured)
    finally:
        context._executor.shutdown(wait=True)

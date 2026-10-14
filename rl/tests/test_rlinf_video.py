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

import imageio
import imageio.v3 as iio
import numpy as np
import pytest

from i4h_rl.backends import rlinf

ROOT = Path(__file__).resolve().parents[2]


@pytest.fixture
def recorder(tmp_path):
    # Import the real methods without requiring GR00T/Ray/Isaac Sim to import
    # the wrapper module. The source is the pinned checkout patched by setup.
    path = ROOT / "third_party/RLinf-a4b6abe/rlinf/envs/wrappers/record_video.py"
    tree = ast.parse(path.read_text())
    methods = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "RecordVideo")
    context = SimpleNamespace(
        _fps=30,
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


def test_encoder_failure_escapes_background_future(recorder, monkeypatch):
    closed = []

    class BrokenWriter:
        def append_data(self, _frame):
            raise OSError("encoder failed")

        def close(self):
            closed.append(True)

    monkeypatch.setattr(imageio, "get_writer", lambda *args, **kwargs: BrokenWriter())
    recorder.render_images = [np.zeros((64, 64, 3), np.uint8)]
    with pytest.raises(RuntimeError, match="encoder failed"):
        recorder.flush_video()
    assert closed and recorder._save_futures[0].done()


def test_evaluation_rejects_corrupt_or_empty_video(tmp_path):
    directory = tmp_path / "video/eval/seed_0"
    directory.mkdir(parents=True)
    path = directory / "0.mp4"
    path.touch()
    assert not rlinf.verify_evaluation_video(tmp_path, Path(sys.executable), os.environ.copy())
    path.write_bytes(b"not an mp4")
    assert not rlinf.verify_evaluation_video(tmp_path, Path(sys.executable), os.environ.copy())


def test_evaluation_accepts_decodable_video(recorder, tmp_path):
    directory = tmp_path / "video/eval/seed_0"
    directory.mkdir(parents=True)
    recorder._save_video([np.zeros((64, 64, 3), np.uint8)], str(directory / "0.mp4"))
    assert rlinf.verify_evaluation_video(tmp_path, Path(sys.executable), os.environ.copy())

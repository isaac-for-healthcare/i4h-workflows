# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import os

import pytest

from i4h_arena.app import _usd_thread_limit


def test_kit_override_is_scoped_and_other_environment_writes_work(monkeypatch):
    monkeypatch.setenv("PXR_WORK_THREAD_LIMIT", "8")
    monkeypatch.setenv("I4H_TEST_SETTING", "before")
    original = type(os.environ).__setitem__
    with pytest.raises(RuntimeError), _usd_thread_limit(1):
        os.environ["PXR_WORK_THREAD_LIMIT"] = "16"  # omni.usd.config bootstrap
        os.environ["I4H_TEST_SETTING"] = "during"
        assert os.environ["PXR_WORK_THREAD_LIMIT"] == "1"
        assert os.environ["I4H_TEST_SETTING"] == "during"
        raise RuntimeError("failed launch")
    assert type(os.environ).__setitem__ is original
    os.environ["PXR_WORK_THREAD_LIMIT"] = "4"
    assert os.environ["PXR_WORK_THREAD_LIMIT"] == "4"


def test_other_scenes_keep_normal_usd_configuration(monkeypatch):
    monkeypatch.delenv("PXR_WORK_THREAD_LIMIT", raising=False)
    original = type(os.environ).__setitem__
    with _usd_thread_limit(None):
        assert type(os.environ).__setitem__ is original
        os.environ["PXR_WORK_THREAD_LIMIT"] = "16"
        assert os.environ["PXR_WORK_THREAD_LIMIT"] == "16"

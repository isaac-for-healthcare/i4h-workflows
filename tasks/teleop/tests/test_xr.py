# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""XR controls must gate simulation/recording and preserve action ordering."""

from types import SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest

from i4h_engine.executor import Engine
from i4h_engine.graph import TaskGraph, node
from i4h_engine.status import WorkflowStatus
from i4h_engine.task import TickContext
from i4h_tasks.basic.testing.fake_scene import FakeActuation, FakeScene
from i4h_tasks.teleop.xr import XR, XRInput


@pytest.fixture
def session(monkeypatch):
    def make(width):
        source = SimpleNamespace(command=None, events=set(), opened=False, closed=False, resets=0)

        def read():
            events, source.events = source.events, set()
            return source.command, events

        source.open = lambda: setattr(source, "opened", True)
        source.close = lambda: setattr(source, "closed", True)
        source.reset = lambda: setattr(source, "resets", source.resets + 1)
        source.read = read
        monkeypatch.setattr("i4h_tasks.teleop.xr.XRInput", lambda *args, **kwargs: source)
        scene = FakeScene(dof=width)
        scene.teleop_config = lambda: object()
        ctx = TickContext(scene=scene, act=FakeActuation(dof=width, action_space="bimanual_pose"))
        engine = Engine(TaskGraph().flow(node(XR())), max_steps=20)
        engine.start(ctx)
        return source, ctx, engine

    return make


@pytest.mark.parametrize("width", [38, 58])
def test_wait_begin_save_and_action_order(session, width):
    source, ctx, engine = session(width)
    source.command = np.arange(width, dtype=np.float32)
    source.events = {"S"}
    engine.tick(ctx)
    assert engine.step == 0 and not engine.advance_requested
    assert engine.status is WorkflowStatus.RUNNING

    source.events = {"B"}
    engine.tick(ctx)
    assert ctx.consume_scene_reset()
    assert engine.step == 0 and not engine.advance_requested
    engine.tick(ctx)
    assert engine.advance_requested
    np.testing.assert_array_equal(ctx.act.raw_actions["robot"], source.command[None, :])
    assert source.resets == 2

    source.events = {"S"}
    engine.tick(ctx)
    assert engine.status is WorkflowStatus.SUCCEEDED
    assert source.closed


def test_discard_requires_new_begin_and_nonempty_demo(session):
    source, ctx, engine = session(38)
    source.events = {"B"}
    engine.tick(ctx)
    ctx.consume_scene_reset()
    source.events = {"S"}
    engine.tick(ctx)
    assert engine.status is WorkflowStatus.RUNNING  # Empty demo cannot be saved.
    source.command = np.zeros(38)
    engine.tick(ctx)
    source.events = {"R"}
    engine.tick(ctx)
    assert ctx.consume_scene_reset()
    source.events = {"S"}
    engine.tick(ctx)
    assert engine.status is WorkflowStatus.RUNNING
    assert not engine.advance_requested
    engine.abort(ctx)
    assert source.closed


@pytest.mark.parametrize("command", [np.zeros(37), np.full(38, np.nan)])
def test_bad_tracking_action_fails_without_advancing(session, command):
    source, ctx, engine = session(38)
    source.events = {"B"}
    engine.tick(ctx)
    ctx.consume_scene_reset()
    source.command = command
    engine.tick(ctx)
    assert engine.status is WorkflowStatus.FAILED
    assert "invalid XR action" in engine.detail
    assert source.closed


@pytest.mark.parametrize("unsubscribe_fails", [False, True])
def test_input_close_releases_device_once_even_when_unsubscribe_fails(unsubscribe_fails):
    source = XRInput.__new__(XRInput)
    source._closed = False
    source._input = Mock()
    source._keyboard = object()
    source._subscription = subscription = object()
    source._device = SimpleNamespace(__exit__=Mock())
    if unsubscribe_fails:
        source._input.unsubscribe_to_keyboard_events.side_effect = RuntimeError("keyboard unavailable")
        with pytest.raises(RuntimeError, match="keyboard unavailable"):
            source.close()
    else:
        source.close()
    source.close()
    source._input.unsubscribe_to_keyboard_events.assert_called_once_with(source._keyboard, subscription)
    source._device.__exit__.assert_called_once_with(None, None, None)
    assert source._subscription is None

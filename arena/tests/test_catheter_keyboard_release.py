# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""CPU regression for Kit event fallback after keyboard polling is unavailable."""

import ast
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("key", ["W", "S", "A", "D", "Q", "E", "Z", "C"])
def test_release_clears_fallback_motion_immediately(monkeypatch, key):
    # Execute the owning event/held-state methods without starting Kit on CPU.
    source = Path(__file__).parents[1] / "i4h_arena/devices/catheter.py"
    tree = ast.parse(source.read_text())
    keyboard = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CatheterKeyboard")
    methods = [
        node
        for node in keyboard.body
        if isinstance(node, ast.FunctionDef) and node.name in {"_active", "_on_keyboard_event"}
    ]
    types = SimpleNamespace(KEY_PRESS=1, KEY_REPEAT=2, KEY_RELEASE=3)
    monkeypatch.setitem(sys.modules, "carb", SimpleNamespace(input=SimpleNamespace(KeyboardEventType=types)))
    namespace = {"time": time, "MOTION_KEYS": frozenset("WSADQEZC")}
    exec(compile(ast.Module(body=methods, type_ignores=[]), str(source), "exec"), namespace)
    device = SimpleNamespace(_key_log=False, _buttons={}, _held_since={}, cfg=SimpleNamespace(key_hold_ttl_s=0.5))
    callback, active = namespace["_on_keyboard_event"], namespace["_active"]
    event = SimpleNamespace(input=key, type=types.KEY_PRESS)
    callback(device, event)
    assert active(device, key)
    event.type = types.KEY_RELEASE
    callback(device, event)
    assert not active(device, key)
    callback(device, event)  # A duplicate release remains harmless.
    assert not device._held_since

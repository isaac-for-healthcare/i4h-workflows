# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The simulator discovers Arena's relocated policy and embodiment packages."""

import importlib.util
from pathlib import Path

import pytest


@pytest.mark.parametrize(
    ("package", "directory"),
    [
        ("isaaclab_arena_g1", "isaaclab_arena_embodiments"),
        ("isaaclab_arena_gr00t", "isaaclab_arena_policies"),
        ("isaaclab_arena_openpi", "isaaclab_arena_policies"),
    ],
)
def test_installed_arena_discovers_relocated_packages(package, directory):
    import isaaclab_arena

    source = Path(isaaclab_arena.__file__).resolve().parents[1]
    spec = importlib.util.find_spec(package)
    assert spec is not None
    assert Path(spec.origin).resolve().is_relative_to(source / directory)

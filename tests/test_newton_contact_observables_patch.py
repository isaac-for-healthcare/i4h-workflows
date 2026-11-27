# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the maintained Lab patch without importing Kit or Newton."""

import os
import re
import subprocess
import textwrap
from pathlib import Path
from types import SimpleNamespace

import pytest

PATCH = Path(__file__).resolve().parents[1] / "third_party/isaaclab_newton19_optional_observables.patch"


def preimage(tmp_path, patch_path):
    patch = patch_path.read_text()
    for section in patch.split("diff --git ")[1:]:
        relative = section.splitlines()[0].split()[0][2:]
        lines = []
        for hunk in re.split(r"(?m)^@@ ", section)[1:]:
            header, body = hunk.split("\n", 1)
            start = int(re.match(r"-(\d+)", header)[1])
            lines.extend(["# padding\n"] * (start - 1 - len(lines)))
            lines.extend(line[1:] for line in body.splitlines(keepends=True) if line.startswith((" ", "-")))
        target = tmp_path / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text("".join(lines))


@pytest.fixture
def manager(tmp_path):
    preimage(tmp_path, PATCH)
    subprocess.run(["git", "apply", str(PATCH)], cwd=tmp_path, check=True)
    source = (tmp_path / "source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py").read_text()
    method = re.search(r"(?ms)^    @classmethod\n    def _step_solver\(.*?(?=^    @classmethod)", source)
    assert method, "the patch must include the complete solver-dispatch method"
    namespace = {}
    exec("from __future__ import annotations\n" + textwrap.dedent(method[0]), namespace)
    return type("Manager", (), {"_step_solver": namespace["_step_solver"]})


def test_legacy_solver_without_contact_observables(manager):
    calls = []

    class LegacySolver:
        def step(self, state_in, state_out, control, contacts, dt):
            calls.append((state_in, state_out, control, contacts, dt))

    manager._solver = LegacySolver()
    manager._contact_observables = None
    args = (object(), object(), object(), object(), 0.01)
    manager._step_solver(*args)
    assert calls == [args]


def test_contact_observables_reach_the_solver(manager):
    calls = []

    class SensorSolver:
        def step(self, *args, observables):
            calls.append((args, observables))

    manager._solver = SensorSolver()
    manager._contact_observables = object()
    args = (object(), object(), object(), object(), 0.01)
    manager._step_solver(*args)
    assert calls == [(args, manager._contact_observables)]


@pytest.mark.parametrize("observables", [None, object()])
def test_solver_type_error_is_not_retried_or_suppressed(manager, observables):
    calls = []

    def step(*args, **kwargs):
        calls.append((args, kwargs))
        raise TypeError("solver internal error")

    manager._solver = SimpleNamespace(step=step)
    manager._contact_observables = observables
    with pytest.raises(TypeError, match="solver internal error"):
        manager._step_solver(None, None, None, None, 0.01)
    assert len(calls) == 1


@pytest.mark.parametrize("state", ["fresh", "previous", "current"])
def test_setup_upgrades_contact_patch_idempotently(tmp_path, state):
    root = PATCH.parents[1]
    full_patch = root / "third_party/isaaclab_newton19_contact_observables.patch"
    preimage(tmp_path, full_patch)
    if state != "fresh":
        subprocess.run(["git", "apply", str(full_patch)], cwd=tmp_path, check=True)
        if state == "previous":
            subprocess.run(["git", "apply", "-R", str(PATCH)], cwd=tmp_path, check=True)
    setup = (root / "third_party/setup.sh").read_text()
    function = re.search(r"(?ms)^apply_patch_once\(\) \{.*?^}\n", setup)[0]
    start = setup.index("  # Upgrade the previous contact patch")
    end = setup.index("\n  # Isaac Lab 3.0 resolves the Pink IK", start)
    script = function + setup[start:end]
    # Reuse the production setup block with this isolated Lab checkout and real patches.
    script = 'set -euo pipefail\nTHIRD_PARTY_DIR="$1"\nISAACLAB_DIR="$2"\nLOG_PREFIX=test\n' + script
    args = ["bash", "-c", script, "patch-setup", str(root / "third_party"), str(tmp_path)]
    # ISAACLAB_DIR is joined to THIRD_PARTY_DIR by setup, so use a relative path.
    args[-1] = os.path.relpath(tmp_path, root / "third_party")
    subprocess.run(args, check=True)
    target = tmp_path / "source/isaaclab_newton/isaaclab_newton/physics/newton_manager.py"
    before = target.read_bytes()
    subprocess.run(args, check=True)
    assert target.read_bytes() == before
    subprocess.run(["git", "apply", "-R", "--check", str(full_patch)], cwd=tmp_path, check=True)

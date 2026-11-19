# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The explicit RL CI plugin rejects missing/skipped required regressions."""

import os
import subprocess
import sys

import pytest

from i4h_rl.pytest_ci import REQUIRED_MODULES


@pytest.mark.parametrize("case", ["complete", "missing", "skipped"])
def test_explicit_ci_plugin_enforces_required_modules(tmp_path, case):
    for index, name in enumerate(sorted(REQUIRED_MODULES)):
        if case == "missing" and index == 0:
            continue
        code = "def test_smoke():\n    assert True\n"
        if case == "skipped" and index == 0:
            code = "import pytest\ndef test_smoke():\n    pytest.skip('injected missing dependency')\n"
        (tmp_path / name).write_text(code)
    result = subprocess.run(
        [sys.executable, "-m", "pytest", "-p", "i4h_rl.pytest_ci", str(tmp_path), "-q"],
        cwd=tmp_path,
        env={**os.environ, "CI": "true"},
        text=True,
        capture_output=True,
    )
    if case == "complete":
        assert result.returncode == 0, result.stdout + result.stderr
    else:
        assert result.returncode == 1, result.stdout + result.stderr
        assert "CI requires the mandatory RL regression modules" in result.stdout

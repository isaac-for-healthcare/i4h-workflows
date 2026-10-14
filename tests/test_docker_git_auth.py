# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Git askpass must restrict the BuildKit credential to GitHub."""

import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("prompt", "expected"),
    [
        ("Username for 'https://github.com': ", "x-access-token"),
        ("Password for 'https://x-access-token@github.com': ", "fake-build-token"),
    ],
)
def test_github_prompt_reads_mounted_secret(tmp_path, prompt, expected):
    helper = _helper(tmp_path, with_secret=True)
    result = subprocess.run([str(helper), prompt], capture_output=True, text=True)
    assert result.returncode == 0
    assert result.stdout.strip() == expected
    assert "fake-build-token" not in result.stderr


def _helper(tmp_path, *, with_secret):
    secret = tmp_path / "github_token"
    if with_secret:
        secret.write_text("fake-build-token")
    helper = tmp_path / "askpass"
    helper.write_text((ROOT / "docker/github-askpass.sh").read_text().replace("/run/secrets/github_token", str(secret)))
    helper.chmod(0o700)
    return helper


def test_other_hosts_never_receive_the_secret(tmp_path):
    helper = _helper(tmp_path, with_secret=True)
    result = subprocess.run(
        [str(helper), "Password for 'https://x-access-token@example.com': "], capture_output=True, text=True
    )
    assert result.returncode != 0
    assert result.stdout == ""
    assert "fake-build-token" not in result.stderr


def test_missing_secret_explains_build_flag(tmp_path):
    helper = _helper(tmp_path, with_secret=False)
    result = subprocess.run([str(helper), "Username for 'https://github.com': "], capture_output=True, text=True)
    assert result.returncode != 0
    assert "--secret id=github_token,env=GH_TOKEN" in result.stderr
    assert result.stdout == ""

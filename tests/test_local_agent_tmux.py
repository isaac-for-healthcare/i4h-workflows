# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the actual tmux adapter with a non-secret canary credential."""

import hashlib
import os
import shutil
import subprocess
import time
import uuid
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
pytestmark = pytest.mark.skipif(shutil.which("tmux") is None, reason="tmux is not installed")


@pytest.fixture(autouse=True)
def isolated_tmux(tmp_path, monkeypatch):
    """Use an independent Bash server, unaffected by the user's tmux config."""
    executable = shutil.which("tmux")
    socket = f"i4h-test-{uuid.uuid4().hex}"
    wrapper_dir = tmp_path / "tmux-bin"
    wrapper_dir.mkdir()
    wrapper = wrapper_dir / "tmux"
    wrapper.write_text(f'#!/bin/sh\nexec {executable} -L {socket} -f /dev/null "$@"\n')
    wrapper.chmod(0o755)
    monkeypatch.setenv("PATH", f"{wrapper_dir}:{os.environ['PATH']}")
    subprocess.run(["tmux", "new-session", "-d", "-s", "bootstrap", "sleep 120"], check=True)
    subprocess.run(["tmux", "set-option", "-g", "default-shell", "/bin/bash"], check=True)
    try:
        yield
    finally:
        subprocess.run(["tmux", "kill-server"], capture_output=True)


def test_key_reaches_shell_without_appearing_in_scrollback(tmp_path):
    session = f"i4h-test-{uuid.uuid4().hex}"
    key = "fake-key-with ' quotes $ and spaces"
    env = dict(
        os.environ,
        I4H_TMUX_SESSION=session,
        I4H_AGENT_REPO_ROOT=str(tmp_path),
        I4H_TMUX_CWD=str(tmp_path),
        I4H_AGENT_VL_API_KEY=key,
    )
    try:
        digest = hashlib.sha256(key.encode()).hexdigest()
        result = subprocess.run(
            [str(ROOT / "local-agent/tmux-shell.sh"), "-c", 'printf %s "$I4H_AGENT_VL_API_KEY" | sha256sum'],
            env=env,
            capture_output=True,
            text=True,
            timeout=30,
        )
        assert result.returncode == 0, result.stderr
        assert digest in result.stdout
        pane = subprocess.check_output(["tmux", "capture-pane", "-p", "-S", "-", "-t", session], text=True)
        assert key not in pane
        assert "export I4H_AGENT_VL_API_KEY=" not in pane
        assert not list(tmp_path.glob(".run/local-agent-tmux.*/env.*"))
    finally:
        subprocess.run(["tmux", "kill-session", "-t", session], capture_output=True)


@pytest.mark.parametrize("termination", ["success", "failure", "signal"])
def test_agent_closes_its_tmux_session(tmp_path, termination):
    agent_dir = tmp_path / "local-agent"
    shutil.copytree(ROOT / "local-agent", agent_dir, ignore=shutil.ignore_patterns("opencode.json", ".run"))
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    curl = bin_dir / "curl"
    curl.write_text("#!/bin/sh\nexit 0\n")
    curl.chmod(0o755)
    opencode = bin_dir / "opencode"
    opencode.write_text(
        "#!/usr/bin/env python3\n"
        "import os, subprocess, sys, time\n"
        "from pathlib import Path\n"
        "subprocess.run([os.environ['TEST_ADAPTER'], '-c', 'true'], check=True)\n"
        "Path(os.environ['TEST_READY']).touch()\n"
        "if os.environ['TEST_END'] == 'signal': time.sleep(60)\n"
        "sys.exit(7 if os.environ['TEST_END'] == 'failure' else 0)\n"
    )
    opencode.chmod(0o755)
    session = f"i4h-test-{uuid.uuid4().hex}"
    ready = tmp_path / "ready"
    env = dict(
        os.environ,
        PATH=f"{bin_dir}:{os.environ['PATH']}",
        I4H_AGENT_PROFILE="nvidia-hosted",
        I4H_AGENT_API_KEY="fake-key",
        I4H_AGENT_REPO_ROOT=str(tmp_path),
        I4H_AGENT_OPENCODE_BIN=str(opencode),
        I4H_TMUX_SESSION=session,
        TEST_ADAPTER=str(agent_dir / "tmux-shell.sh"),
        TEST_READY=str(ready),
        TEST_END=termination,
    )
    process = subprocess.Popen(
        [str(agent_dir / "run.sh"), "agent", "test"], env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True
    )
    try:
        if termination == "signal":
            deadline = time.monotonic() + 15
            while not ready.exists() and process.poll() is None and time.monotonic() < deadline:
                time.sleep(0.05)
            assert ready.exists()
            process.terminate()
        stdout, stderr = process.communicate(timeout=20)
        assert process.returncode == {"success": 0, "failure": 7, "signal": 143}[termination], stdout + stderr
        assert ready.exists()
        assert subprocess.run(["tmux", "has-session", "-t", f"={session}"], capture_output=True).returncode != 0
    finally:
        if process.poll() is None:
            process.kill()
            process.communicate()
        subprocess.run(["tmux", "kill-session", "-t", f"={session}"], capture_output=True)

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``local-agent/vlcheck.py`` reports the requests it made, not its retry budget (NVBug 7048358)."""

from __future__ import annotations

import os
import subprocess
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
VLCHECK = ROOT / "local-agent" / "vlcheck.py"


@pytest.fixture
def endpoint():
    """A chat-completions stub that answers every POST with ``state["status"]`` and counts them."""
    state = {"status": 200, "requests": 0}

    class Handler(BaseHTTPRequestHandler):
        def do_POST(self):  # noqa: N802 - http.server's naming
            self.rfile.read(int(self.headers.get("Content-Length", 0)))
            state["requests"] += 1
            body = b'{"choices": [{"message": {"content": "PASS"}}]}' if state["status"] == 200 else b"{}"
            self.send_response(state["status"])
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, *args):
            pass

    server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        yield f"http://127.0.0.1:{server.server_address[1]}/v1", state
    finally:
        server.shutdown()
        server.server_close()


def _vlcheck(tmp_path, url, *, key=None):
    image = tmp_path / "capture.jpg"
    image.write_bytes(b"\xff\xd8\xff\xd9")
    env = {k: v for k, v in os.environ.items() if not k.startswith("I4H_AGENT_")}
    if key is not None:
        env["I4H_AGENT_VL_API_KEY"] = key
    return subprocess.run(
        [sys.executable, str(VLCHECK), "--image", str(image), "--prompt", "rubric", "--url", url, "--retries", "3"],
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )


@pytest.mark.parametrize("status", [401, 403])
def test_an_auth_error_reports_one_attempt_and_names_the_key(tmp_path, endpoint, status) -> None:
    url, state = endpoint
    state["status"] = status

    result = _vlcheck(tmp_path, url, key="not-a-real-key")

    assert result.returncode == 1
    assert state["requests"] == 1
    assert f"VLCHECK ERROR ({url}) after 1 attempt(s): HTTP Error {status}" in result.stderr
    assert result.stderr.rstrip().endswith("; not retried; check I4H_AGENT_VL_API_KEY")
    assert "not-a-real-key" not in result.stderr


def test_an_auth_error_without_a_key_says_none_is_set(tmp_path, endpoint) -> None:
    url, state = endpoint
    state["status"] = 401

    result = _vlcheck(tmp_path, url)

    assert "after 1 attempt(s)" in result.stderr
    assert "I4H_AGENT_VL_API_KEY is not set (outside the local agent, export it" in result.stderr


def test_another_client_error_is_not_retried(tmp_path, endpoint) -> None:
    url, state = endpoint
    state["status"] = 404

    result = _vlcheck(tmp_path, url)

    assert state["requests"] == 1
    assert result.stderr.rstrip().endswith("after 1 attempt(s): HTTP Error 404: Not Found; not retried")


def test_a_server_error_is_retried_and_reports_every_attempt(tmp_path, endpoint) -> None:
    url, state = endpoint
    state["status"] = 503

    result = _vlcheck(tmp_path, url)

    assert result.returncode == 1
    assert state["requests"] == 3
    assert f"VLCHECK ERROR ({url}) after 3 attempt(s): HTTP Error 503" in result.stderr
    assert "not retried" not in result.stderr


def test_a_verdict_is_printed(tmp_path, endpoint) -> None:
    url, _ = endpoint

    result = _vlcheck(tmp_path, url)

    assert result.returncode == 0
    assert result.stdout.strip() == "PASS"

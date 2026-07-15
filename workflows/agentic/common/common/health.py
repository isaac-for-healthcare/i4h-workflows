# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import json
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


class PolicyHealth:
    def __init__(self, **metadata) -> None:
        self._lock = threading.Lock()
        self._state = "starting"
        self._metadata = {key: value for key, value in metadata.items() if value is not None}

    def set(self, state: str) -> None:
        with self._lock:
            self._state = state

    def set_metadata(self, **metadata) -> None:
        with self._lock:
            self._metadata.update({key: value for key, value in metadata.items() if value is not None})

    def snapshot(self) -> dict:
        with self._lock:
            return {"ok": True, "state": self._state, **self._metadata}


def serve_health(health: PolicyHealth, *, host: str, port: int) -> ThreadingHTTPServer:
    class Handler(BaseHTTPRequestHandler):
        def do_GET(self) -> None:  # noqa: N802
            if self.path not in ("/healthz", "/readyz"):
                self.send_error(404)
                return
            body = json.dumps(health.snapshot()).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def log_message(self, format: str, *args) -> None:
            return

    server = ThreadingHTTPServer((host, port), Handler)
    threading.Thread(target=server.serve_forever, name="agentic-policy-health", daemon=True).start()
    return server

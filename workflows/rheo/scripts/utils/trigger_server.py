# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import asyncio
import logging
import threading

from aiohttp import web
from aiohttp.web import middleware


@middleware
async def cors_middleware(request, handler):
    """Add CORS headers to all responses."""
    if request.method == "OPTIONS":
        response = web.Response()
    else:
        response = await handler(request)

    response.headers["Access-Control-Allow-Origin"] = "*"
    response.headers["Access-Control-Allow-Methods"] = "GET, POST, OPTIONS"
    response.headers["Access-Control-Allow-Headers"] = "Content-Type"
    return response


# ---------------------------------------------------------------------------
# HTTP Trigger Server (for remote policy triggering)
# ---------------------------------------------------------------------------

_trigger_flag = threading.Event()
_reset_flag = threading.Event()

# Robot state: "idle", "running", "complete", "unknown"
_robot_state: str = "unknown"

# Trigger sequence identifier (e.g. "tray_pick_and_place", "cart_push")
_trigger_sequence: str = ""
_trigger_sequence_lock = threading.Lock()
_robot_state_lock = threading.Lock()


def set_robot_state(state: str) -> None:
    """Set the current robot state. Valid states: idle, running, complete, unknown."""
    global _robot_state
    if state not in ("idle", "running", "complete", "unknown"):
        raise ValueError(f"Invalid robot state: {state}")
    with _robot_state_lock:
        _robot_state = state


def get_robot_state() -> str:
    """Get the current robot state."""
    with _robot_state_lock:
        return _robot_state


async def trigger_handler(request):
    """Handle trigger POST request - sets the trigger flag.

    Accepts optional 'sequence' parameter to select which policy to run.
    """
    global _trigger_sequence
    logger = logging.getLogger(__name__)

    # Parse sequence from request body or query params
    sequence = ""
    try:
        if request.content_type == "application/json":
            data = await request.json()
            sequence = str(data.get("sequence", ""))
        elif request.query.get("sequence"):
            sequence = str(request.query.get("sequence"))
    except (ValueError, TypeError):
        pass

    with _trigger_sequence_lock:
        _trigger_sequence = sequence

    _trigger_flag.set()
    logger.info(f"Trigger received via HTTP! Sequence={sequence}")
    return web.json_response(
        {
            "status": "triggered",
            "sequence": sequence,
            "message": f"Policy execution will start (sequence={sequence})",
        }
    )


async def trigger_status_handler(request):
    """Check trigger and robot state."""
    with _trigger_sequence_lock:
        sequence = _trigger_sequence
    return web.json_response(
        {
            "triggered": _trigger_flag.is_set(),
            "robot_state": get_robot_state(),
            "sequence": sequence,
        }
    )


async def trigger_reset_handler(request):
    """Request environment reset via HTTP."""
    _reset_flag.set()
    logger = logging.getLogger(__name__)
    logger.info("Reset requested via HTTP!")
    return web.json_response({"status": "reset_requested", "message": "Environment reset will occur"})


async def _run_trigger_server(host: str, port: int):
    """Run the HTTP trigger server."""
    logger = logging.getLogger(__name__)

    app = web.Application(middlewares=[cors_middleware])
    app.router.add_post("/trigger", trigger_handler)
    app.router.add_get("/status", trigger_status_handler)
    app.router.add_post("/reset", trigger_reset_handler)

    logger.info("Starting HTTP trigger server on %s:%d", host, port)

    runner = web.AppRunner(app)
    await runner.setup()
    site = web.TCPSite(runner, host, port)
    await site.start()

    try:
        await asyncio.Event().wait()
    finally:
        await runner.cleanup()


def start_trigger_server(host: str = "0.0.0.0", port: int = 8081) -> None:
    """Start the HTTP trigger server in a background daemon thread."""

    def _run():
        asyncio.run(_run_trigger_server(host=host, port=port))

    thread = threading.Thread(target=_run, name="http-trigger-server", daemon=True)
    thread.start()


class RemoteTrigger:
    """Remote trigger that can be activated via HTTP POST to /trigger or /reset.

    Supports a ``sequence`` parameter (string) to select which policy to run.
    The sequence name maps directly to a policy config key defined by the caller.
    """

    def __init__(self, host: str = "0.0.0.0", port: int = 8081):
        self.host = host
        self.port = port
        _trigger_flag.clear()
        _reset_flag.clear()
        set_robot_state("idle")
        start_trigger_server(host=host, port=port)

    def consume_trigger(self) -> str | None:
        """Consume the pending trigger if one exists.

        Atomically checks and clears the trigger flag.

        Returns:
            The sequence name if a trigger was pending, ``None`` otherwise.
        """
        if _trigger_flag.is_set():
            with _trigger_sequence_lock:
                seq = _trigger_sequence
            _trigger_flag.clear()
            return seq
        return None

    def check_reset(self) -> bool:
        """Check if reset was requested and clear the flag.

        Returns:
            True if reset was requested, False otherwise.
        """
        if _reset_flag.is_set():
            _reset_flag.clear()
            return True
        return False

    def clear(self):
        """Clear all pending flags without changing robot state."""
        _trigger_flag.clear()
        _reset_flag.clear()

    @staticmethod
    def set_state(state: str) -> None:
        """Set the robot state. Valid states: idle, running, complete, unknown."""
        set_robot_state(state)

    @staticmethod
    def get_state() -> str:
        """Get the current robot state."""
        return get_robot_state()

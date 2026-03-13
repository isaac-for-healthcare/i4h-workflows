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


import logging
import re
import threading
import time
from typing import Any, Dict, List, Optional

import requests
from agents.base_agent import Agent


class RobotControlAgent(Agent):
    """
    Robot control agent.
    Triggers the robot by sending a POST request to the trigger endpoint.

    Supports two modes:
    1. Direct request: Called via process_request when user says "start"
       - If pending trigger exists (from event), uses that sequence
       - Otherwise determines sequence based on video source mode
    2. Event-driven: Subscribes to "robot_trigger" events from message bus
       - Caches the trigger request, awaits user confirmation ("start")
       - Pending trigger expires after timeout (default 60s)
    """

    def __init__(self, settings_path: str, response_handler=None, get_video_source_mode=None, message_bus=None):
        super().__init__(settings_path, response_handler, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)

        # Video source mode callback
        self.get_video_source_mode = get_video_source_mode

        # Get trigger endpoint from config, default to localhost:8081
        self._trigger_endpoint: str = self.agent_settings.get("control_endpoint", "http://localhost:8081/trigger")
        self._timeout_s: float = float(self.agent_settings.get("command_timeout_s", 5.0))

        # Pending trigger timeout (seconds) - how long to wait for user confirmation
        self._pending_timeout_s: float = float(self.agent_settings.get("pending_trigger_timeout_s", 60.0))

        # Pending trigger from event (expires after timeout).
        # Guarded by _pending_lock because the message-bus event thread
        # writes to it while the main thread reads/clears it.
        self._pending_trigger: Optional[Dict[str, Any]] = None
        self._pending_lock = threading.Lock()

        # Subscribe to robot_trigger events from message bus
        if message_bus:
            message_bus.subscribe_to_event("robot_trigger", self._on_robot_trigger_event)
            self._logger.info("RobotControlAgent subscribed to 'robot_trigger' events")

    def _on_robot_trigger_event(self, message):
        """Cache robot_trigger event for user confirmation.

        The trigger is cached and will be executed when user says "start".
        Expires after pending_trigger_timeout_s seconds.
        """
        try:
            payload = message.payload if hasattr(message, "payload") else message
            sequence = str(payload.get("sequence", ""))
            source = payload.get("source", "unknown")

            self._logger.info(
                f"Received robot_trigger event from {source} with sequence={sequence} - awaiting user confirmation"
            )

            # Cache the request (will be used when user says "start")
            with self._pending_lock:
                self._pending_trigger = {
                    "sequence": sequence,
                    "source": source,
                    "timestamp": time.time(),
                }

        except Exception as e:
            self._logger.error(f"Error handling robot_trigger event: {e}", exc_info=True)

    def _get_pending_trigger(self) -> Optional[Dict[str, Any]]:
        """Get pending trigger if valid (not expired).

        Returns:
            Pending trigger dict if valid, None if expired or not set.
        """
        with self._pending_lock:
            if self._pending_trigger is None:
                return None

            elapsed = time.time() - self._pending_trigger["timestamp"]
            if elapsed > self._pending_timeout_s:
                self._logger.info(
                    f"Pending trigger from {self._pending_trigger['source']} expired "
                    f"({elapsed:.1f}s > {self._pending_timeout_s}s)"
                )
                self._pending_trigger = None
                return None

            return self._pending_trigger

    def _clear_pending_trigger(self):
        """Clear the pending trigger."""
        with self._pending_lock:
            if self._pending_trigger:
                self._logger.debug(f"Cleared pending trigger from {self._pending_trigger['source']}")
                self._pending_trigger = None

    def _trigger_robot(self, sequence: Optional[str] = None) -> Dict[str, Any]:
        """Send trigger request to robot endpoint.

        Args:
            sequence: Optional sequence name (e.g. "tray_pick_and_place", "cart_push")

        Returns:
            Response dict with status
        """
        try:
            self._logger.info(f"Triggering robot at {self._trigger_endpoint} with sequence={sequence}")

            if sequence is not None:
                response = requests.post(self._trigger_endpoint, json={"sequence": sequence}, timeout=self._timeout_s)
            else:
                response = requests.post(self._trigger_endpoint, timeout=self._timeout_s)

            if response.ok:
                self._logger.info(f"Robot triggered successfully: {response.text}")
                return {"success": True, "response": response.text}
            else:
                error_msg = f"Failed to trigger robot: HTTP {response.status_code}"
                self._logger.error(error_msg)
                return {"success": False, "error": error_msg}

        except requests.Timeout:
            error_msg = f"Timeout waiting for robot response from {self._trigger_endpoint}"
            self._logger.error(error_msg)
            return {"success": False, "error": error_msg}
        except requests.RequestException as e:
            error_msg = f"Failed to connect to robot: {e}"
            self._logger.error(error_msg)
            return {"success": False, "error": error_msg}

    _DISAPPROVAL_KEYWORDS = {"no", "stop", "cancel", "reject", "decline", "abort", "deny", "negative"}

    def _is_disapproval(self, text: str) -> bool:
        """Return True if the user text expresses disapproval / rejection."""
        words = set(re.findall(r"\w+", text.lower()))
        return bool(words & self._DISAPPROVAL_KEYWORDS)

    def process_request(
        self,
        text: str,
        chat_history: List[List[Optional[str]]],
        visual_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """
        Process the robot control request by sending a POST to the trigger endpoint.

        Only triggers if there is a pending trigger from monitor agents.
        If no pending trigger exists, returns a message indicating no action needed.
        If the user disapproves, the pending trigger is cleared without firing.
        """
        # Check for pending trigger from monitor agents
        pending = self._get_pending_trigger()

        if not pending:
            self._logger.info("No pending trigger request to approve")
            return {
                "name": "RobotControlAgent",
                "response": "There is no pending robot action to approve at this time.",
                "command": "none",
            }

        sequence = pending["sequence"]
        source = pending["source"]

        # Handle user disapproval — clear the pending trigger without firing
        if self._is_disapproval(text):
            self._logger.info(f"User rejected pending trigger from {source} with sequence={sequence}")
            self._clear_pending_trigger()
            return {
                "name": "RobotControlAgent",
                "response": "Robot action has been cancelled.",
                "command": "cancelled",
            }

        # User approved — execute the pending trigger
        self._logger.info(f"User approved pending trigger from {source} with sequence={sequence}")
        self._clear_pending_trigger()

        result = self._trigger_robot(sequence=sequence)

        if result.get("success"):
            return {
                "name": "RobotControlAgent",
                "response": "Command successfully sent to the robot.",
                "command": "trigger",
            }
        else:
            return {
                "name": "RobotControlAgent",
                "response": result.get("error", "Unknown error"),
                "command": "trigger",
                "error": result.get("error"),
            }

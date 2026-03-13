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


import json
import logging
import threading
import time

from agents.base_agent import Agent


class PushCartMonitorAgent(Agent):
    """
    Background agent that uses VLM to monitor if the surgical tray is placed on the case cart.
    When detected for consecutive frames, prompts the user to send the robot to move the cart.
    """

    def __init__(
        self,
        settings_path,
        response_handler=None,
        agent_key=None,
        get_latest_frame=None,
        message_bus=None,
        get_video_source_mode=None,
        get_session_id=None,
    ):
        super().__init__(settings_path, response_handler, agent_key=agent_key, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)

        self.get_latest_frame = get_latest_frame
        self.get_video_source_mode = get_video_source_mode
        self.get_session_id = get_session_id
        self._last_analyzed_frame = None

        # Configuration
        self.consecutive_threshold = self.agent_settings["consecutive_threshold"]
        self.check_interval = self.agent_settings["check_interval_seconds"]
        self.prompt_message = self.agent_settings["prompt_message"]
        self._analysis_prompt = self.agent_settings["prompt_presets"][0]["text"]
        self.robot_trigger_sequence = self.agent_settings.get("robot_trigger_sequence", "cart_push")

        # State tracking
        self.consecutive_detected_count = 0
        self.prompt_sent = False  # Only send prompt once per detection episode
        self._last_mode = None
        self._last_session_id = None

        # Background thread control
        self.stop_event = threading.Event()

        if self.get_latest_frame is not None:
            self.thread = threading.Thread(target=self._background_loop, daemon=True)
            self.thread.start()
            self._logger.info("PushCartMonitorAgent started.")
        else:
            self.thread = None
            self._logger.warning("PushCartMonitorAgent: No get_latest_frame provided, running in passive mode")

    def _reset_state(self):
        """Reset agent state when session or mode changes."""
        self.prompt_sent = False
        self.consecutive_detected_count = 0

    def _is_mode_valid(self, mode: str) -> bool:
        """Check if the current mode is valid for the agent."""
        current_mode = self.get_video_source_mode()

        if current_mode != mode:
            # Only log when the current mode is no longer valid
            if self._last_mode == mode:
                self._logger.info(f"Video source changed from '{self._last_mode}' to '{current_mode}'. Pausing.")
            return False
        return True

    def _reset_if_mode_or_session_change(self) -> None:
        """Check video source mode and session, reset state if needed."""
        current_mode = self.get_video_source_mode()
        current_session_id = self.get_session_id()

        reset_needed = False
        # Reset state when switching to a new mode
        if self._last_mode != current_mode:
            self._logger.info(f"Video source changed to '{current_mode}'")
            reset_needed = True
            self._last_mode = current_mode
        # Reset state when switching to a new session
        if self._last_session_id != current_session_id:
            self._logger.info("New session detected")
            reset_needed = True
            self._last_session_id = current_session_id

        if reset_needed:
            self._reset_state()

    def _check_tray_on_cart(self, image_b64: str) -> bool | None:
        """Query VLM to check if surgical tray is placed on the case cart.

        Returns:
            True if tray is on cart, False if not, None if error.
        """
        try:
            response = self.stream_image_response(
                prompt=self._analysis_prompt,
                image_b64=image_b64,
                grammar=self.grammar,
                temperature=0.0,
                display_output=False,
            )

            # Parse the structured response
            if isinstance(response, str):
                result = json.loads(response)
            else:
                result = response

            tray_on_cart = result.get("tray_on_cart")
            self._logger.info(f"VLM response: tray_on_cart={tray_on_cart}")
            return tray_on_cart

        except json.JSONDecodeError as e:
            self._logger.warning(f"Failed to parse VLM response as JSON: {e}")
            return None
        except Exception as e:
            self._logger.error(f"VLM query failed: {e}", exc_info=True)
            return None

    def _background_loop(self):
        """Monitor frames and check if surgical tray is on the cart using VLM."""
        while not self.stop_event.is_set():
            # Check for session and mode changes
            self._reset_if_mode_or_session_change()

            if not self._is_mode_valid("operating_room"):
                time.sleep(self.check_interval)
                continue

            if self.prompt_sent:
                time.sleep(self.check_interval)
                continue

            # Get the most recent frame directly
            frame_data = self.get_latest_frame("operating_room")

            # Skip if no frame or same frame already analyzed
            if frame_data is None or frame_data is self._last_analyzed_frame:
                time.sleep(self.check_interval)
                continue

            # Validate frame data
            if not isinstance(frame_data, str) or len(frame_data) < 1000:
                self._logger.debug("Invalid or empty frame data, skipping.")
                time.sleep(self.check_interval)
                continue

            self._last_analyzed_frame = frame_data

            # Query VLM to check if tray is on cart
            tray_on_cart = self._check_tray_on_cart(frame_data)

            if tray_on_cart is True:
                self.consecutive_detected_count += 1
                self._logger.debug(
                    f"Tray on cart detected ({self.consecutive_detected_count}/{self.consecutive_threshold})"
                )

                # Check if we've hit the threshold
                if self.consecutive_detected_count >= self.consecutive_threshold:
                    self._send_prompt()
                    self.prompt_sent = True
            else:
                # Reset count if tray is not on cart
                self.consecutive_detected_count = 0

            time.sleep(self.check_interval)

    def _send_prompt(self):
        """Send prompt to user and trigger robot with sequence=2."""
        self._logger.warning("🛒 Surgical tray detected on cart for consecutive frames - triggering robot")

        # Send alert to UI
        alert_payload = {
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            "source": "cart_monitor",
            "description": self.prompt_message,
            "surgical_phase": "Perioperative",
        }
        try:
            # Send alert to UI
            self.message_bus.publish_event(sender_agent=self.agent_name, event_type="tray_alert", payload=alert_payload)

            # Trigger robot with configured sequence (push cart policy)
            trigger_payload = {
                "sequence": self.robot_trigger_sequence,
                "source": "cart_monitor",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            self.message_bus.publish_event(
                sender_agent=self.agent_name, event_type="robot_trigger", payload=trigger_payload
            )
            self._logger.info(f"Published robot_trigger event with sequence={self.robot_trigger_sequence}")

        except Exception as e:
            self._logger.error(f"Error publishing cart events: {e}")

    def process_request(self, user_text, chat_history, visual_info=None):
        """Handle direct queries about cart status."""
        return {
            "name": "PushCartMonitorAgent",
            "response": f"Cart Monitor: Tracking surgical tray placement on case cart. "
            f"Consecutive detections: {self.consecutive_detected_count}/{self.consecutive_threshold}. "
            f"Prompt sent: {self.prompt_sent}",
        }

    def stop(self):
        """Stop the background thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self._logger.info("Stopping PushCartMonitorAgent")
            self.thread.join(timeout=5)

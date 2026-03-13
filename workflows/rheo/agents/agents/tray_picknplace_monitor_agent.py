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


class TrayPickNPlaceMonitorAgent(Agent):
    """
    Background agent that monitors surgical tray status from PeriOpAnnotationAgent.
    Triggers an alert when tray has been missing for consecutive checks.
    """

    def __init__(
        self,
        settings_path,
        response_handler=None,
        agent_key=None,
        peri_op_annotation_agent=None,
        message_bus=None,
        get_video_source_mode=None,
        get_session_id=None,
    ):
        super().__init__(settings_path, response_handler, agent_key=agent_key, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)

        # Reference to PeriOpAnnotationAgent to read its annotations
        self.peri_op_annotation_agent = peri_op_annotation_agent
        self.get_video_source_mode = get_video_source_mode
        self.get_session_id = get_session_id

        # Configuration
        self.consecutive_threshold = self.agent_settings["consecutive_threshold"]
        self.check_interval = self.agent_settings["check_interval_seconds"]
        self.alert_message = self.agent_settings["alert_message"]
        self.robot_trigger_sequence = self.agent_settings.get("robot_trigger_sequence", "tray_pick_and_place")

        # State tracking
        self.alert_sent = False  # Only send alert once per missing episode
        self._last_mode = None  # Track mode changes to reset alert state
        self._last_session_id = None  # Track session changes to reset state on reconnect
        self._baseline_count = 0  # Ignore annotations before this index after mode switch

        # Background thread control
        self.stop_event = threading.Event()

        if self.peri_op_annotation_agent is not None:
            self.thread = threading.Thread(target=self._background_loop, daemon=True)
            self.thread.start()
            self._logger.info("TrayPickNPlaceMonitorAgent started.")
        else:
            self.thread = None
            self._logger.warning(
                "TrayPickNPlaceMonitorAgent: No peri_op_annotation_agent provided, running in passive mode"
            )

    def _reset_state(self):
        """Reset agent state when session or mode changes."""
        self.alert_sent = False

        all_annotations = self.peri_op_annotation_agent.get_annotations()
        self._baseline_count = len(all_annotations)

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
        """Check video source mode and reset state if needed.

        Returns:
            True if should continue processing, False if should skip this iteration.
        """
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

    def _get_tray_status(self, annotation) -> bool:
        """Extract surgical_tray status from annotation description.

        Returns:
            True if tray is visible, False if not visible, None if unknown.
        """
        description = annotation.get("description", "")
        if isinstance(description, str):
            # Description is json.dumps of analysis
            analysis = json.loads(description)
            return analysis.get("surgical_tray")

    def _background_loop(self):
        """Monitor annotations and check for missing tray."""
        while not self.stop_event.is_set():
            # Check for session and mode changes
            self._reset_if_mode_or_session_change()

            if not self._is_mode_valid("operating_room"):
                time.sleep(self.check_interval)
                continue

            if self.alert_sent:
                time.sleep(self.check_interval)  # won't send alert unless reset
                continue

            # Get the latest annotations from PeriOpAnnotationAgent
            all_annotations = self.peri_op_annotation_agent.get_annotations()

            # Only consider annotations after the baseline (post mode-switch)
            annotations = all_annotations[self._baseline_count :]
            current_count = len(annotations)

            if current_count < self.consecutive_threshold:
                time.sleep(self.check_interval)
                continue

            # Check the last N annotations
            recent = annotations[-self.consecutive_threshold :]

            missing_count = sum(1 for ann in recent if self._get_tray_status(ann) is False)

            # Check if we've hit the threshold
            if missing_count >= self.consecutive_threshold:
                self._send_alert()
                self.alert_sent = True

            time.sleep(self.check_interval)

    def _send_alert(self):
        """Send alert to UI and trigger robot with sequence=1."""
        self._logger.warning("🚨 Surgical tray missing for consecutive checks - triggering robot")

        if self.message_bus:
            # Send alert to UI
            alert_payload = {
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                "source": "tray_monitor",
                "description": self.alert_message,
                "surgical_phase": "Perioperative",
            }
            try:
                # Send alert to UI
                self.message_bus.publish_event(
                    sender_agent=self.agent_name, event_type="tray_alert", payload=alert_payload
                )

                # Trigger robot with configured sequence (pick and place policy)
                trigger_payload = {
                    "sequence": self.robot_trigger_sequence,
                    "source": "tray_monitor",
                    "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
                }
                self.message_bus.publish_event(
                    sender_agent=self.agent_name, event_type="robot_trigger", payload=trigger_payload
                )
                self._logger.info(f"Published robot_trigger event with sequence={self.robot_trigger_sequence}")

            except Exception as e:
                self._logger.error(f"Error publishing tray events: {e}")

    def process_request(self, user_text, chat_history, visual_info=None):
        """Handle direct queries about tray status."""
        return {
            "name": "TrayPickNPlaceMonitorAgent",
            "response": f"Tray Monitor: Currently tracking surgical tray visibility. "
            f"Alert threshold: {self.consecutive_threshold} consecutive missing detections. "
            f"Alert sent: {self.alert_sent}",
        }

    def stop(self):
        """Stop the background thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self._logger.info("Stopping TrayPickNPlaceMonitorAgent")
            self.thread.join(timeout=5)

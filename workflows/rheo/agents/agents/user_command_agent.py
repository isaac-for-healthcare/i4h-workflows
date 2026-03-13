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
import time
from typing import Any, Dict, List, Optional

from agents.base_agent import Agent


class UserCommandAgent(Agent):
    """Interpret natural-language user commands and map them to robot actions.

    When a user says something like "bring the surgical tray" or "push the
    cart to the table", this agent uses the LLM to determine which robot
    policy (sequence number) to run and publishes a ``robot_trigger`` event
    on the message bus.  The existing :class:`RobotControlAgent` picks up
    that event and waits for the user to say "start" before executing.
    """

    def __init__(self, settings_path: str, response_handler=None, message_bus=None):
        super().__init__(settings_path, response_handler, message_bus=message_bus)
        self._logger = logging.getLogger(__name__)

    def process_request(
        self,
        text: str,
        chat_history: List[List[Optional[str]]],
        visual_info: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Interpret user text, publish a robot_trigger event, and respond."""
        prompt = self.generate_prompt(text, chat_history)
        raw = self.stream_response(prompt, grammar=self.grammar, display_output=False)

        # Parse the LLM's structured output
        try:
            result = json.loads(raw)
        except (json.JSONDecodeError, TypeError):
            self._logger.warning("Failed to parse LLM output: %s", raw)
            return {
                "name": "UserCommandAgent",
                "response": "Sorry, I could not understand that command. "
                "Try something like 'bring the surgical tray' or 'push the cart'.",
            }

        sequence = result.get("sequence")
        description = result.get("description", "")

        if sequence is None:
            return {
                "name": "UserCommandAgent",
                "response": description or "I could not determine a robot action from your request.",
            }

        # Publish a robot_trigger event (same pattern as the monitor agents)
        if self.message_bus:
            trigger_payload = {
                "sequence": sequence,
                "source": "user_command",
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime()),
            }
            self.message_bus.publish_event(
                sender_agent=self.agent_name,
                event_type="robot_trigger",
                payload=trigger_payload,
            )
            self._logger.info("Published robot_trigger event with sequence=%s (from user command)", sequence)

        confirmation = description or f"Preparing robot action (sequence {sequence})."
        return {
            "name": "UserCommandAgent",
            "response": f"{confirmation} Say 'start' to confirm.",
        }

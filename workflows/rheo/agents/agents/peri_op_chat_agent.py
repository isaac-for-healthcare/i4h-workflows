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

from agents.base_agent import Agent


class PeriOpChatAgent(Agent):
    """
    Chat agent for operating room scene conversations.
    Uses the agent_prompt as the system prompt and supports visual chat.

    If a PeriOpAnnotationAgent instance is provided via ``peri_op_annotation_agent``,
    the most recently detected scene context is automatically prepended to the user
    prompt, including visible instruments, the surgical tray, and case cart.
    """

    def __init__(self, settings_path, response_handler, peri_op_annotation_agent=None):
        super().__init__(settings_path, response_handler)
        self.peri_op_annotation_agent = peri_op_annotation_agent

    def _get_annotation_context(self):
        """
        Return the full scene context from the most recent PeriOpAnnotationAgent annotation.

        Returns a dict with:
            'instruments' : list[str]  - visible surgical instruments
            'surgical_tray': bool      - whether a surgical tray is visible
            'case_cart': bool         - whether a case cart is visible
        Returns an empty dict if no annotation is available yet.
        """
        if self.peri_op_annotation_agent is None:
            return {}
        try:
            annotations = self.peri_op_annotation_agent.get_annotations()
            if not annotations:
                return {}
            latest = annotations[-1]
            instruments = list(latest.get("tools", []))
            analysis = {}
            description = latest.get("description", "")
            if description:
                try:
                    analysis = json.loads(description)
                except Exception:
                    pass
            return {
                "instruments": instruments,
                "surgical_tray": bool(analysis.get("surgical_tray", False)),
                "case_cart": bool(analysis.get("case_cart", False)),
            }
        except Exception as e:
            self._logger.warning(f"Could not read annotation context: {e}")
        return {}

    def _build_user_text(self, text):
        """
        Prepend scene-context hint to the user text using the latest annotation.

        Builds an item list from:
          - visible surgical instruments
          - "Surgical Tray"  (when surgical_tray is True)
          - "Case Cart"  (when case_cart is True)

        Format:
          "If the user's intent is to understand the objects in the image,
           here are visible surgical tools: {items}. {text}"
        """
        ctx = self._get_annotation_context()
        if not ctx:
            self._logger.info("[ToolContext] No annotation available yet; sending raw user text.")
            return text

        items = list(ctx.get("instruments", []))
        if ctx.get("surgical_tray"):
            items.append("Surgical Tray")
        if ctx.get("case_cart"):
            items.append("Case Cart")

        if not items:
            self._logger.info("[ToolContext] Annotation present but no items detected; sending raw user text.")
            return text

        item_str = ", ".join(items)
        prefix = (
            f"If the user's intent is to understand the objects in the image, "
            f"here are visible surgical tools: {item_str}."
        )
        self._logger.info(f"[ToolContext] Prepending to user prompt: {prefix}")
        return f"{prefix} {text}"

    def process_request(self, text, chat_history, visual_info=None):
        """
        Process a user request with optional image data.

        When a PeriOpAnnotationAgent is available, the user text is automatically
        enriched with the latest detected tool information before being sent.

        Args:
            text: The user's query/message
            chat_history: List of previous conversation turns
            visual_info: Optional dict with 'image_b64' for the current frame

        Returns:
            dict with 'name' and 'response' keys
        """
        try:
            self._logger.debug("Starting PeriOpChatAgent process_request")
            self._logger.debug(f"Input text: {text}")

            if not visual_info:
                visual_info = {}

            image_b64 = visual_info.get("image_b64", None)

            enriched_text = self._build_user_text(text)
            prompt = self.generate_prompt(enriched_text, chat_history)

            if image_b64:
                self._logger.debug("Received image data, calling stream_image_response.")
                response = self.stream_image_response(prompt=prompt, image_b64=image_b64, temperature=0.0)
            else:
                self._logger.debug("No image data, calling stream_response.")
                response = self.stream_response(prompt=prompt, temperature=0.0)

            return {"name": "PeriOpChatAgent", "response": response}

        except Exception as e:
            self._logger.error(f"Error in PeriOpChatAgent.process_request: {e}", exc_info=True)
            return {"name": "PeriOpChatAgent", "response": f"Error: {str(e)}"}


__all__ = ["PeriOpChatAgent"]

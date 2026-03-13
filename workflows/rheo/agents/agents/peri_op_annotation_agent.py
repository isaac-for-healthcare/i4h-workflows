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
import os
import threading
import time

from agents.base_agent import Agent


class PeriOpAnnotationAgent(Agent):
    """
    Background annotation agent for operating room / peri-operative scenes.
    Continuously processes frames and generates structured annotations.
    """

    def __init__(
        self,
        settings_path,
        response_handler=None,
        agent_key=None,
        procedure_start_str=None,
        get_latest_frame=None,
        on_annotation_callback=None,
        get_video_source_mode=None,
        get_session_id=None,
    ):
        super().__init__(settings_path, response_handler, agent_key=agent_key)
        self._logger = logging.getLogger(__name__)

        self.get_latest_frame = get_latest_frame
        self.on_annotation_callback = on_annotation_callback
        self.time_step = self.agent_settings.get("time_step_seconds", 10)
        self.get_video_source_mode = get_video_source_mode
        self.get_session_id = get_session_id
        self._last_analyzed_frame = None
        self._last_mode = None
        self._last_session_id = None

        # If procedure_start_str is not provided, create one
        if procedure_start_str is None:
            procedure_start_str = time.strftime("%Y_%m_%d__%H_%M_%S", time.localtime())

        self.procedure_start_str = procedure_start_str
        self.procedure_start = time.time()

        # Output directory setup
        base_output_dir = self.agent_settings.get("annotation_output_dir", "annotations")
        self.procedure_folder = os.path.join(base_output_dir, f"procedure_{self.procedure_start_str}")
        os.makedirs(self.procedure_folder, exist_ok=True)

        # Annotation output file
        self.annotation_filepath = os.path.join(self.procedure_folder, "peri_op_annotations.json")
        self._logger.info(f"PeriOpAnnotationAgent output: {self.annotation_filepath}")

        # Annotation storage
        self.annotations = []

        # Load prompt from config
        prompt_presets = self.agent_settings.get("prompt_presets", [])
        self._analysis_prompt = prompt_presets[0]["text"]

        # Background thread control
        self.stop_event = threading.Event()

        # Start background loop if get_latest_frame is provided
        if self.get_latest_frame is not None:
            self.thread = threading.Thread(target=self._background_loop, daemon=True)
            self.thread.start()
            self._logger.info(f"PeriOpAnnotationAgent background thread started (interval={self.time_step}s).")
        else:
            self.thread = None
            self._logger.info("PeriOpAnnotationAgent initialized without get_latest_frame (on-demand mode).")

    def _reset_state(self):
        """Reset episode state on mode or session change."""
        self._last_analyzed_frame = None

    def _is_mode_valid(self, mode: str) -> bool:
        """Check if the current mode is valid for the agent."""
        current_mode = self.get_video_source_mode()

        if current_mode != mode:
            if self._last_mode == mode:
                self._logger.info(f"Video source changed from '{self._last_mode}' to '{current_mode}'. Pausing.")
            return False
        return True

    def _reset_if_mode_or_session_change(self):
        """Detect mode/session changes and reset state when either occurs."""
        current_mode = self.get_video_source_mode() if self.get_video_source_mode else None
        current_session_id = self.get_session_id() if self.get_session_id else None

        reset_needed = False
        if self._last_mode != current_mode:
            self._logger.info(f"Video source changed to '{current_mode}'")
            reset_needed = True
            self._last_mode = current_mode
        if current_session_id is not None and self._last_session_id != current_session_id:
            self._logger.info("New session detected")
            reset_needed = True
            self._last_session_id = current_session_id

        if reset_needed:
            self._reset_state()

    def _background_loop(self):
        """Continuously process the latest frame."""
        # Poll frequently until the first frame is processed, then use the
        # configured time_step cadence to avoid unnecessary VLM calls.
        poll_interval = 1.0

        while not self.stop_event.is_set():
            try:
                self._reset_if_mode_or_session_change()

                # Check if we should process based on current mode
                if self.get_video_source_mode:
                    if not self._is_mode_valid("operating_room"):
                        time.sleep(poll_interval)
                        continue

                # Get the most recent frame directly
                frame_data = self.get_latest_frame("operating_room")

                # Skip if no frame or same frame already analyzed.
                # Sleep in 1-second increments so a reconnect / new frame wakes
                # us up quickly instead of blocking for the full poll_interval.
                if frame_data is None or frame_data is self._last_analyzed_frame:
                    waited = 0.0
                    while waited < poll_interval and not self.stop_event.is_set():
                        time.sleep(1.0)
                        waited += 1.0
                        latest = self.get_latest_frame("operating_room")
                        if latest is not None and latest is not self._last_analyzed_frame:
                            break
                    continue

                # Validate frame data
                if not isinstance(frame_data, str) or len(frame_data) < 1000:
                    self._logger.debug("Invalid or empty frame data, skipping.")
                    time.sleep(poll_interval)
                    continue

                # Generate annotation
                annotation = self._generate_annotation(frame_data)
                if annotation:
                    # Mark as analyzed only after a successful annotation so that
                    # a failed attempt doesn't lock out the same frame for time_step seconds.
                    self._last_analyzed_frame = frame_data
                    self.annotations.append(annotation)
                    self.append_json_to_file(annotation, self.annotation_filepath)
                    self._logger.info(f"✓ Generated annotation: {annotation.get('analysis', {})}")

                    # Settle into the normal cadence only after the first success.
                    poll_interval = self.time_step

                    # Notify via callback if set (sends to UI)
                    if self.on_annotation_callback:
                        try:
                            self.on_annotation_callback(annotation)
                        except Exception as e:
                            self._logger.error(f"Error in annotation callback: {e}")
                else:
                    self._logger.warning("Annotation generation failed; will retry on next frame.")

            except Exception as e:
                self._logger.error(f"Error in background loop: {e}", exc_info=True)

    def _generate_annotation(self, image_b64):
        """Generate an annotation from an image frame."""
        timestamp_str = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        elapsed = time.time() - self.procedure_start

        # Send to LLM
        try:
            response = self.stream_image_response(
                prompt=self._analysis_prompt,
                image_b64=image_b64,
                grammar=self.grammar,
                temperature=0.0,
                display_output=False,
            )
            analysis = json.loads(response) if isinstance(response, str) else response
        except Exception as e:
            self._logger.error(f"Analysis failed: {e}", exc_info=True)
            return None

        return {
            "timestamp": timestamp_str,
            "elapsed_time_seconds": elapsed,
            "source": "operating_room",
            "tools": analysis.get("visible_instruments", []),
            "anatomy": ["N/A"],
            "surgical_phase": "Perioperative",
            "description": json.dumps(analysis),
        }

    def process_request(self, user_text, chat_history, visual_info=None):
        return {
            "name": "PeriOpAnnotationAgent",
            "response": "PeriOpAnnotationAgent runs in the background and generates annotations.",
        }

    def stop(self):
        """Stop the background thread."""
        self.stop_event.set()
        if self.thread and self.thread.is_alive():
            self._logger.info("Stopping PeriOpAnnotationAgent background thread.")
            self.thread.join(timeout=5)

    def get_annotations(self):
        """Return all generated annotations."""
        return self.annotations

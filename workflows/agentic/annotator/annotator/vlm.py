# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import base64
import io
import json
import re
from typing import Any

import numpy as np
from annotator.records import FrameBundle
from openai import OpenAI
from PIL import Image


class VLMVerifier:
    def __init__(
        self,
        *,
        base_url: str,
        api_key: str,
        model: str,
        temperature: float = 0.0,
        max_tokens: int = 512,
    ) -> None:
        self._client = OpenAI(base_url=base_url, api_key=api_key)
        self._model = model
        self._temperature = temperature
        self._max_tokens = max_tokens

    def annotate(self, *, task_description: str, frames: list[FrameBundle], context: str) -> dict[str, Any]:
        content: list[dict[str, Any]] = [{"type": "text", "text": _prompt(task_description, frames, context)}]
        for frame in frames:
            for camera_name, image in frame.cameras.items():
                content.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": _image_data_url(image)},
                    }
                )
                content.append(
                    {
                        "type": "text",
                        "text": f"Image above: camera={camera_name}, sampled_frame={frame.index}.",
                    }
                )

        response = self._client.chat.completions.create(
            model=self._model,
            messages=[{"role": "user", "content": content}],
            temperature=self._temperature,
            max_tokens=self._max_tokens,
        )
        raw = response.choices[0].message.content or ""
        parsed = _parse_json_object(raw)
        parsed.setdefault("raw_response", raw)
        parsed.setdefault("label", "bad")
        parsed.setdefault("success", parsed.get("label") == "good")
        return parsed


def _prompt(task_description: str, frames: list[FrameBundle], context: str) -> str:
    frame_list = ", ".join(str(frame.index) for frame in frames)
    return f"""You are a strict visual verifier for robot manipulation episodes.

Task description:
{task_description}

Input context:
{context}

You will receive camera images sampled from the same attempt. Sampled frame indices: {frame_list}.
Evaluate the images against the task description above. The task description is the source of truth:
it may ask for completed task success, or it may ask for a preflight visibility/sanity check where
objects only need to be visible.

Return only one JSON object with this schema:
{{
  "label": "good" or "bad",
  "success": true or false,
  "confidence": number from 0.0 to 1.0,
  "reasoning": "brief explanation",
  "evidence": ["short visual observations"]
}}

Mark "good" only when the requested visual condition is visible or strongly implied. If the images are
ambiguous or do not satisfy the requested condition, mark "bad" with lower confidence.
"""


def _image_data_url(image: np.ndarray) -> str:
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim != 3 or arr.shape[-1] != 3:
        raise ValueError(f"expected RGB image with shape (H, W, 3), got {arr.shape}")
    buffer = io.BytesIO()
    Image.fromarray(arr, mode="RGB").save(buffer, format="JPEG", quality=90)
    encoded = base64.b64encode(buffer.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _parse_json_object(raw: str) -> dict[str, Any]:
    try:
        value = json.loads(raw)
        return value if isinstance(value, dict) else {"raw_response": raw}
    except json.JSONDecodeError:
        pass

    match = re.search(r"\{.*\}", raw, flags=re.DOTALL)
    if not match:
        return {"raw_response": raw}
    try:
        value = json.loads(match.group(0))
    except json.JSONDecodeError:
        return {"raw_response": raw}
    return value if isinstance(value, dict) else {"raw_response": raw}

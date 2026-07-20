#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Send bridge capture image(s) to the local vision-language model for a scene-validation verdict.

Used by the i4h-* skills to do real *visual* validation: the coding agent (a non-vision
model) captures the bridge viewport, then calls this to have the VL model judge the scene.

Usage:
  python3 local-agent/vlcheck.py --image <viewport.jpg> [--image <pov.jpg> ...] --prompt "<rubric>"

Endpoint/model come from env (defaults match the local-agent VL config):
  I4H_AGENT_VL_BASE_URL (default https://inference-api.nvidia.com)
  I4H_AGENT_VL_MODEL (default nvidia/nvidia/nemotron-nano-12b-v2-vl)
  I4H_AGENT_VL_API_KEY (optional bearer token; required for the hosted default)

Prints the model's verdict to stdout; exits non-zero on transport error.
"""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
import time
import urllib.error
import urllib.request


def default_vl_url() -> str:
    base = os.environ.get("I4H_AGENT_VL_BASE_URL", "https://inference-api.nvidia.com").rstrip("/")
    return base + "/v1"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--image", action="append", required=True, help="capture image path (repeatable)")
    ap.add_argument("--prompt", required=True, help="validation rubric / question for the VL model")
    ap.add_argument("--url", default=default_vl_url())
    ap.add_argument("--model", default=os.environ.get("I4H_AGENT_VL_MODEL", "nvidia/nvidia/nemotron-nano-12b-v2-vl"))
    ap.add_argument("--max-tokens", type=int, default=500)
    ap.add_argument("--timeout", type=int, default=120, help="per-attempt request timeout (s)")
    ap.add_argument(
        "--retries",
        type=int,
        default=int(os.environ.get("I4H_AGENT_VL_RETRIES", "3")),
        help="max attempts (the hosted VL endpoint occasionally wedges a single request)",
    )
    a = ap.parse_args()

    content = [{"type": "text", "text": a.prompt}]
    for img in a.image:
        b64 = base64.b64encode(open(img, "rb").read()).decode()
        content.append({"type": "image_url", "image_url": {"url": "data:image/jpeg;base64," + b64}})

    payload = {
        "model": a.model,
        "messages": [{"role": "user", "content": content}],
        "max_tokens": a.max_tokens,
        "temperature": 0,
    }
    headers = {"Content-Type": "application/json"}
    api_key = os.environ.get("I4H_AGENT_VL_API_KEY")
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"

    url = a.url.rstrip("/") + "/chat/completions"

    # The hosted VL endpoint occasionally wedges a single request (no response, or a
    # transient 429/5xx). That stall is rare, so retrying almost always succeeds and
    # turns a "VL-DEFERRED" (visual check skipped) into a real verdict. We do NOT retry
    # deterministic client errors (400/401/403/404) — those won't fix themselves.
    attempts = max(1, a.retries)
    last_err = None
    for attempt in range(attempts):
        req = urllib.request.Request(url, data=json.dumps(payload).encode(), headers=headers)
        try:
            resp = json.load(urllib.request.urlopen(req, timeout=a.timeout))
            print(resp["choices"][0]["message"]["content"])
            return 0
        except urllib.error.HTTPError as exc:
            last_err = exc
            if exc.code in (400, 401, 403, 404):  # deterministic — don't retry
                break
        except Exception as exc:  # noqa: BLE001 — timeout / transport: retryable
            last_err = exc
        if attempt + 1 < attempts:
            print(f"VLCHECK retry {attempt + 1}/{attempts - 1} after: {last_err}", file=sys.stderr)
            time.sleep(2)
    print(f"VLCHECK ERROR ({a.url}) after {attempts} attempt(s): {last_err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())

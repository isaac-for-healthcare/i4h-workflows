# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""A ``PolicyServer`` stand-in for smoking the deployment client without a robot or a GPU.

Not a test (no ``test_`` prefix, never collected). It answers ``ping``,
``server_info``, ``reset`` and ``get_action`` on the wire format of
``client/protocol.py``, serves a ``server_info`` shaped like
``server.py:served_info()`` for a bimanual or a left-only checkpoint, validates
every observation against its own ``required_observation_keys`` the way the
policy's strict input check would, and returns a ``(1, 32, D)`` chunk per
action key that continues the current state with a small ramp. Needs pyzmq
and msgpack, so it runs in the client environment or the task venv::

    python tasks/gr00t_n17_tactile/tests/fake_policy_server.py --port 5599 [--hands left|both] [--no-force] [--cascade]
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.client.protocol import pack, unpack  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.tactile.interface import DEFORM_SHAPE, FINGERTIPS, FORCE_AXES  # noqa: E402

ACTION_HORIZON = 32
FORCE_WINDOW = 16
DIMS = {"left_arm": 7, "right_arm": 7, "left_hand": 22, "right_hand": 22}
LANGUAGE_KEY = "annotation.human.task_description"


def served_info(hands: str, force: bool, cascade: bool) -> dict[str, Any]:
    """``server_info`` for the chosen topology, in the schema of ``server.py:served_info()``."""
    if hands == "left":
        fingertips = [name for name in FINGERTIPS if name.startswith("left_")]
        video = ["high", "right_wrist_view"]
        state = ["left_arm", "right_arm", "left_hand"]
    else:
        fingertips = list(FINGERTIPS)
        video = ["high", "left_wrist_view", "right_wrist_view"]
        state = ["left_arm", "right_arm", "left_hand", "right_hand"]
    groups = {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 36], "right_hand": [36, 58]}
    required = {
        "video": video,
        "state": state,
        "language": [LANGUAGE_KEY],
        "tactile": [f"{name}_deform" for name in fingertips],
    }
    if force:
        required["tactile_force"] = ["force"]
    return {
        "task": "gr00t_n17_tactile/h2_sharpa_tactile",
        "embodiment": "new_embodiment",
        "video_modality_keys": video,
        "state_groups": {key: groups[key] for key in state},
        "action_horizon": ACTION_HORIZON,
        "execution_steps": 16,
        "control_hz": 30.0,
        "tactile": {
            "enabled": True,
            "source": "none",
            "fingertips": fingertips,
            "num_tokens": len(fingertips),
            "feature_dim": 512,
            "force_window": FORCE_WINDOW if force else 0,
            "delay_aug": True,
            "refresh": "per-step",
        },
        "cascade": {"enabled": bool(cascade), "fast_loops": 2},
        "checkpoint": {
            "class": "Gr00tN1d7Tactile",
            "use_tactile": True,
            "use_tactile_force": bool(force),
            "delay_aug": True,
            "tau_split": 0.5,
            "force_delay_max": 16,
            "num_tactile_tokens": len(fingertips),
        },
        "required_observation_keys": required,
    }


def validate(obs: dict[str, Any], info: dict[str, Any]) -> str | None:
    """The first problem with ``obs`` against ``required_observation_keys``, or None."""
    required = info["required_observation_keys"]
    extra = set(obs) - set(required)
    if extra:
        return f"unexpected modalities {sorted(extra)}"
    columns = FORCE_AXES * len(info["tactile"]["fingertips"])
    for modality, keys in required.items():
        got = obs.get(modality)
        if not isinstance(got, dict):
            return f"missing modality {modality}"
        if set(got) != set(keys):
            return f"{modality} keys {sorted(got)} != {sorted(keys)}"
        for key, value in got.items():
            if modality == "language":
                if not (isinstance(value, list) and len(value) == 1 and len(value[0]) == 1):
                    return f"language.{key} is not [[str]]"
                continue
            shape = tuple(np.shape(value))
            if modality == "tactile_force":
                if shape != (1, FORCE_WINDOW, columns):
                    return f"tactile_force.{key} shape {shape} != {(1, FORCE_WINDOW, columns)}"
            elif modality == "tactile":
                if shape != (1, 1, *DEFORM_SHAPE):
                    return f"tactile.{key} shape {shape}"
            elif modality == "state":
                if shape != (1, 1, DIMS[key]):
                    return f"state.{key} shape {shape} != {(1, 1, DIMS[key])}"
            elif len(shape) != 5 or shape[:2] != (1, 1) or shape[-1] != 3:
                return f"video.{key} shape {shape}"
    return None


def make_chunk(obs: dict[str, Any], info: dict[str, Any]) -> dict[str, np.ndarray]:
    """Continue the current state with a small ramp, one ``(1, 32, D)`` array per served state key."""
    chunk = {}
    ramp = 0.001 * np.arange(ACTION_HORIZON, dtype=np.float32)[:, None]
    for key in info["required_observation_keys"]["state"]:
        current = np.asarray(obs["state"][key], dtype=np.float32).reshape(-1)
        chunk[key] = (current[None, :] + ramp)[None]
    return chunk


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="PolicyServer stand-in for client smokes.")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=5599)
    ap.add_argument("--hands", choices=["left", "both"], default="both")
    ap.add_argument("--force", action=argparse.BooleanOptionalAction, default=True)
    ap.add_argument("--cascade", action="store_true", help="advertise the slow/fast cascade (same-plan smokes)")
    args = ap.parse_args(argv)

    import zmq

    info = served_info(args.hands, args.force, args.cascade)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind(f"tcp://{args.host}:{args.port}")
    print(f"fake policy server on {args.host}:{args.port} hands={args.hands} force={args.force} cascade={args.cascade}")
    calls = 0
    try:
        while True:
            request = unpack(socket.recv())
            endpoint = request.get("endpoint")
            data = request.get("data") or {}
            if endpoint == "ping":
                reply: Any = {"status": "ok"}
            elif endpoint == "server_info":
                reply = info
            elif endpoint == "reset":
                print("reset")
                reply = {}
            elif endpoint == "get_action":
                calls += 1
                obs = data.get("observation") or {}
                problem = validate(obs, info)
                if problem:
                    print(f"get_action #{calls} REJECTED: {problem}", flush=True)
                    reply = {"error": problem}
                else:
                    options = data.get("options")
                    print(f"get_action #{calls} ok options={options}", flush=True)
                    reply = [make_chunk(obs, info), {"options": options}]
            elif endpoint == "kill":
                socket.send(pack({"status": "ok"}))
                break
            else:
                reply = {"error": f"Unknown endpoint: {endpoint}"}
            socket.send(pack(reply))
    except KeyboardInterrupt:
        pass
    finally:
        socket.close(linger=0)
        context.term()
    print(f"fake policy server stopped after {calls} get_action calls")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

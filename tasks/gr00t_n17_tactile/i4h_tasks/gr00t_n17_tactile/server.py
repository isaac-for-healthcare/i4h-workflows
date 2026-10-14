# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""GR00T N1.7 tactile backend.

Serves ``gr00t_n17_tactile/*`` tasks to a real Unitree H2 over Isaac-GR00T's own
``PolicyServer`` (ZMQ request/reply with msgpack, ``gr00t/policy/server_client.py``
in the n1.7 release), the protocol that the deployment clients shipped with
Isaac-GR00T already speak. It does not subclass
:class:`i4h_common.server.PolicyServer`: that bus carries simulator
observations, and no Arena scene produces fingertip tactile data today. When
one does, the adapter is one class that wraps the same policy; everything in
this module stays as it is.

What differs from ``tasks/gr00t_n17/server.py``:

* the observation carries ten fingertip deformation maps and a causal 6-axis
  force window next to video, state and language;
* the policy may re-read the live tactile stream once per denoising step
  (``tactile_refresh: per-step``) through the provider the patch series adds
  to ``Gr00tPolicy.get_action``;
* an optional slow/fast cascade re-plans the tail of each chunk with fresher
  tactile data;
* a ``server_info`` endpoint tells the client what the checkpoint expects, so
  the client carries no copy of the training configuration.

Structure-first draft: options, manifest resolution, ``server_info`` and the
live tactile provider are complete; loading the policy and binding the server
raise until the GR00T patch series lands (``patches/README.md``).
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np

from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile")

DEFAULT_TASK = "gr00t_n17_tactile/h2_sharpa_tactile"
DEFAULT_MODALITY_CONFIG = "config"
TACTILE_SOURCES = ("live", "none")
TACTILE_REFRESH = ("per-step", "per-chunk")


@dataclass(frozen=True)
class ServeOptions:
    """Everything the command line can say; ``None`` means "as the manifest says"."""

    task: str = DEFAULT_TASK
    checkpoint: str = ""
    host: str = "0.0.0.0"
    port: int = 5555
    tactile_source: str = "live"
    tactile_host: str = "127.0.0.1"
    tactile_port: int = 7779
    tactile_refresh: str | None = None
    cascade: bool | None = None
    cascade_fast_loops: int | None = None
    #: Load the tactile checkpoint, then run it with the tactile path switched
    #: off: the vision-only ablation on the same weights.
    disable_tactile: bool = False


def _register_modalities(module: str) -> None:
    """Import a sibling modality config for its registration side effect.

    Checked against the modules that exist, so a typo in a manifest names the
    alternatives instead of raising a bare ImportError.
    """
    here = os.path.dirname(__file__)
    if not os.path.isfile(os.path.join(here, f"{module}.py")):
        available = sorted(name[:-3] for name in os.listdir(here) if name.startswith("config") and name.endswith(".py"))
        raise ValueError(f"no modality config {module!r} in this stack; have {available}")
    importlib.import_module(f"{__package__}.{module}")


def inference_settings(spec: Any, options: ServeOptions) -> dict[str, Any]:
    """The manifest's ``model.inference`` block with the command line laid over it."""
    declared = dict(spec.model.get("inference") or {})
    settings = {
        "tactile_refresh": declared.get("tactile_refresh", "per-step"),
        "cascade": bool(declared.get("cascade", False)),
        "cascade_fast_loops": int(declared.get("cascade_fast_loops", 2)),
    }
    if options.tactile_refresh is not None:
        settings["tactile_refresh"] = options.tactile_refresh
    if options.cascade is not None:
        settings["cascade"] = options.cascade
    if options.cascade_fast_loops is not None:
        settings["cascade_fast_loops"] = options.cascade_fast_loops
    if settings["tactile_refresh"] not in TACTILE_REFRESH:
        raise ValueError(f"tactile_refresh must be one of {TACTILE_REFRESH}, got {settings['tactile_refresh']!r}")
    return settings


def server_info(spec: Any, options: ServeOptions) -> dict[str, Any]:
    """What a client learns when it connects.

    Derived from the manifest and the options rather than from constants in the
    client, so the client of a bimanual checkpoint and the client of a
    single-hand one are the same program.
    """
    model = spec.model
    tactile = model.get("tactile") or {}
    force = tactile.get("force") or {}
    inference = inference_settings(spec, options)
    num_tokens = int(tactile.get("num_tokens", len(FINGERTIPS)))
    return {
        "task": options.task,
        "embodiment": spec.embodiment,
        "video_modality_keys": list(spec.cameras),
        "state_groups": dict(spec.observation.get("state_groups") or {}),
        "action_horizon": model.get("action_horizon"),
        "execution_steps": model.get("execution_steps"),
        "control_hz": model.get("control_hz"),
        "tactile": {
            "enabled": not options.disable_tactile,
            "source": options.tactile_source,
            "fingertips": list(FINGERTIPS[:num_tokens]),
            "feature_dim": tactile.get("feature_dim"),
            "force_window": int(force.get("window", 0)) if force.get("enabled") else 0,
            "delay_aug": bool((tactile.get("delay_aug") or {}).get("enabled", False)),
            "refresh": inference["tactile_refresh"],
        },
        "cascade": {"enabled": inference["cascade"], "fast_loops": inference["cascade_fast_loops"]},
    }


def build_tactile_provider(spec: Any, options: ServeOptions) -> Callable[[], dict[str, np.ndarray] | None] | None:
    """The callable the policy invokes for fresh tactile inputs, or ``None``.

    ``None`` when the tactile path is off, in which case the policy runs on
    vision and state alone. Otherwise every call returns the newest complete
    frame as model inputs, or ``None`` to keep the previous step's inputs when
    a fingertip has not reported yet.
    """
    if options.disable_tactile or options.tactile_source == "none":
        return None
    from i4h_tasks.gr00t_n17_tactile.tactile.sharpa import SharpaLiveSource

    window = server_info(spec, options)["tactile"]["force_window"]
    source = SharpaLiveSource(options.tactile_host, options.tactile_port, force_window=max(window, 1))

    def provider() -> dict[str, np.ndarray] | None:
        frame = source.latest()
        if frame is None:
            return None
        inputs = {"deform_maps": frame.deform[np.newaxis, :, np.newaxis], "timestamp": np.float64(frame.timestamp)}
        if window:
            inputs["tactile_force"] = source.force_window(window)[np.newaxis]
        for hand, joints in (frame.hand_joints or {}).items():
            inputs[f"joint_angles_{hand}"] = joints[np.newaxis]
        return inputs

    return provider


def load_policy(spec: Any, options: ServeOptions) -> Any:
    """``Gr00tPolicy`` for the checkpoint, with the tactile options applied."""
    _register_modalities(str(spec.model.get("modality_config") or DEFAULT_MODALITY_CONFIG))
    if not options.checkpoint:
        raise ValueError(f"{options.task}: no published checkpoint; pass --checkpoint")
    raise NotImplementedError(
        "Gr00tPolicy with tactile options lands with the GR00T patch series; "
        "see tasks/gr00t_n17_tactile/patches/README.md"
    )


def serve(options: ServeOptions) -> int:
    from i4h_common.training import task_spec

    spec = task_spec(options.task)
    info = server_info(spec, options)
    logger.info("server_info: %s", json.dumps(info))
    provider = build_tactile_provider(spec, options)
    policy = load_policy(spec, options)
    # Isaac-GR00T's PolicyServer(policy, host, port) with ping / reset / get_action /
    # get_modality_config, plus server_info and the per-step provider wired into
    # get_action's options; the cascade wraps get_action when enabled.
    raise NotImplementedError(f"binding {type(policy).__name__} with provider={provider is not None} is not wired yet")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t-n17-tactile-server", description="Serve a tactile GR00T N1.7 checkpoint."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="task id, e.g. gr00t_n17_tactile/h2_sharpa_tactile")
    parser.add_argument("--checkpoint", default="", help="finetuned checkpoint directory (no published default)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--tactile-source", choices=TACTILE_SOURCES, default="live")
    parser.add_argument("--tactile-host", default="127.0.0.1", help="host of the robot-side tactile bridge")
    parser.add_argument("--tactile-port", type=int, default=7779)
    parser.add_argument("--tactile-refresh", choices=TACTILE_REFRESH, default=None, help="overrides the manifest")
    parser.add_argument("--cascade", action=argparse.BooleanOptionalAction, default=None, help="overrides the manifest")
    parser.add_argument("--cascade-fast-loops", type=int, default=None)
    parser.add_argument("--disable-tactile", action="store_true", help="vision-only ablation of a tactile checkpoint")
    parser.add_argument("--print-server-info", action="store_true", help="resolve the manifest, print the info, stop")
    parser.add_argument("--verbose", action="store_true")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO, format="[gr00t-n17-tactile] %(message)s")
    options = ServeOptions(
        task=args.task,
        checkpoint=args.checkpoint,
        host=args.host,
        port=args.port,
        tactile_source=args.tactile_source,
        tactile_host=args.tactile_host,
        tactile_port=args.tactile_port,
        tactile_refresh=args.tactile_refresh,
        cascade=args.cascade,
        cascade_fast_loops=args.cascade_fast_loops,
        disable_tactile=args.disable_tactile,
    )
    if args.print_server_info:
        from i4h_common.training import task_spec

        try:
            spec = task_spec(options.task)
        except KeyError as exc:
            print(f"error: {exc}", file=sys.stderr)
            return 2
        print(json.dumps(server_info(spec, options), indent=2))
        return 0
    return serve(options)


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

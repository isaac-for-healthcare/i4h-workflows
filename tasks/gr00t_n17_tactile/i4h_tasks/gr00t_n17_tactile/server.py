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

The model comes from the patched Isaac-GR00T checkout that
``third_party/setup.sh tasks/gr00t_n17_tactile`` creates (``patches/README.md``);
the frozen encoders it builds are imported from their public checkouts, which
:func:`~.tactile.sharpa.import_public_sources` puts on ``sys.path`` first.
"""

from __future__ import annotations

import argparse
import importlib
import json
import logging
import os
import sys
import time
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS, HANDS

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
    device: str = "cuda"
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


#: Seconds between two warnings about an incomplete live stream.
STREAM_WARNING_INTERVAL = 5.0
#: Seconds the live stream gets to deliver every fingertip before serving starts.
STREAM_READY_TIMEOUT = 2.0


def build_tactile_provider(
    spec: Any, options: ServeOptions, fingertips: Iterable[str] = FINGERTIPS
) -> Callable[[], dict[str, np.ndarray] | None] | None:
    """The callable the policy invokes for fresh tactile inputs, or ``None``.

    ``None`` when the tactile path is off, in which case the policy runs on
    vision and state alone. Otherwise the live source is opened for exactly the
    checkpoint's ``fingertips`` and must deliver every one of them within
    :data:`STREAM_READY_TIMEOUT`; after that every call returns the newest
    complete frame as model inputs, or ``None`` (with a warning at most every
    :data:`STREAM_WARNING_INTERVAL`) to keep the previous step's inputs.

    Only the deform maps and the hand joints sampled with them are returned.
    The force window stays at request time: force freshness comes from the
    client's observation (per fast pass in the cascade), which is how the
    checkpoints were deployed.
    """
    if options.disable_tactile or options.tactile_source == "none":
        return None
    from i4h_tasks.gr00t_n17_tactile.tactile.sharpa import SharpaLiveSource

    source = SharpaLiveSource(options.tactile_host, options.tactile_port, fingertips=fingertips)
    source.wait_ready(STREAM_READY_TIMEOUT)
    logger.info("live tactile: %s on tcp://%s:%d", list(source.fingertips), options.tactile_host, options.tactile_port)
    last_warning = [float("-inf")]

    def provider() -> dict[str, np.ndarray] | None:
        frame = source.latest()
        if frame is None:
            now = time.monotonic()
            if now - last_warning[0] >= STREAM_WARNING_INTERVAL:
                last_warning[0] = now
                logger.warning("live tactile incomplete, keeping the previous frame; missing %s", source.missing())
            return None
        inputs = {"deform_maps": frame.deform[np.newaxis, :, np.newaxis]}
        for hand, joints in (frame.hand_joints or {}).items():
            inputs[f"joint_angles_{hand}"] = joints[np.newaxis]
        return inputs

    return provider


def refresh_fields(inputs: dict[str, Any], hands: Iterable[str] = HANDS) -> dict[str, Any]:
    """The part of a provider frame that the model re-reads per denoising step.

    The deform maps and the hand joints of the hands the checkpoint uses, so
    the fingertip kinematics see the pose the maps were taken at; joints of a
    hand the checkpoint does not have, and anything else, are dropped.
    """
    wanted = {"deform_maps", *(f"joint_angles_{hand}" for hand in hands)}
    return {key: value for key, value in inputs.items() if key in wanted}


def checkpoint_config(checkpoint: str) -> dict[str, Any]:
    """The ``config.json`` a checkpoint directory was saved with."""
    directory = Path(checkpoint).expanduser()
    if not directory.is_dir():
        raise FileNotFoundError(f"no checkpoint directory at {directory}")
    config = directory / "config.json"
    if not config.is_file():
        raise FileNotFoundError(f"{directory} has no config.json; is it a GR00T checkpoint?")
    return json.loads(config.read_text())


def checkpoint_hands(saved: dict[str, Any]) -> list[str]:
    """The hands a checkpoint's saved config has a URDF for, in canonical order."""
    return [hand for hand in HANDS if saved.get(f"sharpa_{hand}_urdf_path")]


def urdf_overrides(saved: dict[str, Any], assets: dict[str, Any]) -> dict[str, str]:
    """The ``Gr00tPolicy`` URDF overrides for a checkpoint.

    Only the URDFs the checkpoint has are overridden with the pinned copies: a
    single-hand checkpoint has the other hand's path unset, and giving it a
    second hand would change the model.
    """
    return {f"sharpa_{hand}_urdf_path": str(assets[f"{hand}_urdf"]) for hand in checkpoint_hands(saved)}


def checkpoint_fingertips(tactile_keys: Iterable[str]) -> list[str]:
    """Canonical fingertip names of a checkpoint's ``<hand>_<finger>_deform`` tactile keys, in its order."""
    names = []
    for key in tactile_keys:
        name = key.removesuffix("_deform")
        if name not in FINGERTIPS:
            raise ValueError(f"unexpected tactile key {key!r}; the fingertips are {FINGERTIPS}, each with '_deform'")
        names.append(name)
    return names


def policy_fingertips(policy: Any) -> list[str]:
    configs = getattr(policy, "modality_configs", {})
    if "tactile" not in configs:
        raise ValueError("live tactile needs a tactile checkpoint; the loaded one has no tactile modality")
    return checkpoint_fingertips(configs["tactile"].modality_keys)


def checkpoint_facts(policy: Any) -> dict[str, Any]:
    """What the loaded checkpoint says about itself, for ``server_info``."""
    model = getattr(policy, "model", None)
    config = getattr(model, "config", None)
    return {
        "class": type(model).__name__ if model is not None else None,
        "use_tactile": bool(getattr(config, "use_tactile", False)),
        "use_tactile_force": bool(getattr(config, "use_tactile_force", False)),
        "delay_aug": bool(getattr(config, "delay_aug", False)),
        "tau_split": float(getattr(config, "tau_split", 0.5)),
        "force_delay_max": int(getattr(config, "force_delay_max", 0)),
        "num_tactile_tokens": int(getattr(config, "num_tactile_tokens", 0)),
    }


def load_policy(spec: Any, options: ServeOptions) -> Any:
    """``Gr00tPolicy`` for the checkpoint, with the tactile options applied.

    The checkpoint's saved config names the encoder file and the URDFs of the
    machine it was trained on; they are overridden with this repository's
    pinned copies. Only the URDFs the checkpoint has are overridden, because a
    single-hand checkpoint has one of them unset and giving it a second hand
    would change the model. The finetuned checkpoint carries the frozen encoder
    tensors, so the encoder file only serves to build the module.
    """
    _register_modalities(str(spec.model.get("modality_config") or DEFAULT_MODALITY_CONFIG))
    if not options.checkpoint:
        raise ValueError(f"{options.task}: no published checkpoint; pass --checkpoint")
    from i4h_tasks.gr00t_n17_tactile.tactile.sharpa import import_public_sources, resolve_assets

    checkpoint = str(Path(options.checkpoint).expanduser())
    saved = checkpoint_config(checkpoint)
    import_public_sources()
    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.policy.gr00t_policy import Gr00tPolicy

    overrides: dict[str, str] = {}
    if saved.get("use_tactile"):
        hands = checkpoint_hands(saved)
        assets = resolve_assets(
            spec.model.get("tactile") or {}, ["deform_encoder", *(f"{hand}_urdf" for hand in hands)]
        )
        overrides = {"sharpa_encoder_ckpt_path": str(assets["deform_encoder"]), **urdf_overrides(saved, assets)}
    elif not options.disable_tactile and options.tactile_source != "none":
        raise ValueError(f"{checkpoint} is a vision-only checkpoint; pass --tactile-source none")
    logger.info("loading %s", checkpoint)
    policy = Gr00tPolicy(
        model_path=checkpoint,
        embodiment_tag=EmbodimentTag.resolve(str(spec.model.get("embodiment_tag", "new_embodiment"))),
        device=options.device,
        **overrides,
    )
    if options.disable_tactile:
        head = getattr(policy.model, "action_head", None)
        if head is None or not hasattr(head, "use_tactile"):
            raise ValueError("--disable-tactile needs a tactile checkpoint")
        # Same weights, no tactile influence: the SAT encoder and the tactile
        # cross-attention are bypassed, which is the vision-only ablation.
        head.use_tactile = False
    return policy


def served_info(info: dict[str, Any], policy: Any, options: ServeOptions) -> dict[str, Any]:
    """The manifest's ``server_info`` corrected by what the loaded checkpoint actually expects.

    Camera views, state groups, fingertips, force window and delay augmentation
    come from the checkpoint's processor and model config, so the client of a
    single-hand checkpoint learns five fingertips and one wrist view without the
    manifest having to say so. ``required_observation_keys`` lists every key of
    every modality the policy's strict input check demands.
    """
    configs = getattr(policy, "modality_configs", {})
    config = getattr(getattr(policy, "model", None), "config", None)
    out = json.loads(json.dumps(info))
    if "video" in configs:
        out["video_modality_keys"] = list(configs["video"].modality_keys)
    if "state" in configs:
        groups = out.get("state_groups") or {}
        out["state_groups"] = {key: groups[key] for key in configs["state"].modality_keys if key in groups}
    fingertips = checkpoint_fingertips(configs["tactile"].modality_keys) if "tactile" in configs else []
    use_force = bool(getattr(config, "use_tactile_force", False))
    out["tactile"].update(
        fingertips=fingertips,
        num_tokens=len(fingertips),
        force_window=int(getattr(config, "force_window", 0)) if use_force else 0,
        delay_aug=bool(getattr(config, "delay_aug", False)),
    )
    out["checkpoint"] = checkpoint_facts(policy)
    out["required_observation_keys"] = {
        name: list(configs[name].modality_keys)
        for name in ("video", "state", "language", "tactile", "tactile_force")
        if name in configs
    }
    if options.disable_tactile:
        out["notes"] = [
            "--disable-tactile switches the tactile path off inside the model only; the observation must still "
            "carry the tactile and tactile_force keys, which the policy's strict input check requires for a "
            "tactile checkpoint."
        ]
    return out


def with_live_tactile(policy: Any, provider: Callable[[], dict[str, np.ndarray] | None], refresh: str) -> Callable:
    """``get_action`` with the live stream wired into every denoising step.

    The action head calls the injected provider once per step and merges what it
    returns onto the request's inputs. The provider delivers the checkpoint's
    fingertips in the checkpoint's order (see :func:`build_tactile_provider`).
    ``per-step`` snapshots the stream on every call; ``per-chunk`` snapshots
    once per request and reuses it. When the stream has no complete frame the
    previous frame of this request is kept, or the request's own tactile data
    before the first one.
    """
    import torch

    hands = [hand for hand in HANDS if any(name.startswith(f"{hand}_") for name in policy_fingertips(policy))]
    parameter = next(policy.model.parameters())
    state: dict[str, Any] = {"frame": None}

    def to_model(frame: dict[str, np.ndarray]) -> dict[str, Any]:
        return {
            key: torch.from_numpy(np.ascontiguousarray(np.asarray(value, dtype=np.float32))).to(
                device=parameter.device, dtype=parameter.dtype
            )
            for key, value in frame.items()
        }

    def model_provider() -> dict[str, Any]:
        if refresh == "per-chunk" and state["frame"] is not None:
            return state["frame"]
        inputs = provider()
        if inputs is None:
            return state["frame"] or {}
        state["frame"] = to_model(refresh_fields(inputs, hands))
        return state["frame"]

    def get_action(observation: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        state["frame"] = None
        return policy.get_action(observation, options={**(options or {}), "tactile_input_provider": model_provider})

    return get_action


def with_cascade(policy: Any, get_action: Callable, fast_loops: int) -> Callable:
    """``get_action`` that runs the slow/fast cascade inside the model.

    The model caches the backbone output and the early denoising state across
    ``fast_loops`` requests and re-plans the tail of each chunk with the
    request's fresh tactile data. That is in distribution only for a checkpoint
    trained with delay augmentation, which is checked here.
    """
    config = getattr(getattr(policy, "model", None), "config", None)
    if not getattr(config, "use_tactile", False) or not getattr(config, "delay_aug", False):
        raise ValueError("the cascade needs a checkpoint trained with tactile data and delay augmentation")
    loops = max(1, int(fast_loops))
    force_delay_max = int(getattr(config, "force_delay_max", 0))
    if loops - 1 > force_delay_max:
        # Across the fast passes the tactile data drifts up to loops-1 observations
        # ahead of the cached backbone output; training covered force_delay_max.
        logger.warning(
            "cascade_fast_loops - 1 = %d exceeds the checkpoint's force_delay_max = %d: the fast tail may see "
            "tactile data further ahead of the vision than training covered",
            loops - 1,
            force_delay_max,
        )

    def cascade_get_action(observation: dict[str, Any], options: dict[str, Any] | None = None) -> Any:
        return get_action(observation, options={**(options or {}), "cascade": True, "cascade_fast_loops": loops})

    return cascade_get_action


def serve(options: ServeOptions) -> int:
    from i4h_common.training import task_spec

    spec = task_spec(options.task)
    settings = inference_settings(spec, options)
    live = not options.disable_tactile and options.tactile_source != "none"
    if settings["cascade"] and live:
        raise ValueError(
            "the cascade refreshes tactile data from every client observation and does not read a live "
            "stream; pass --tactile-source none with --cascade"
        )
    if settings["cascade"] and options.disable_tactile:
        raise ValueError("--cascade and --disable-tactile exclude each other")
    policy = load_policy(spec, options)
    info = served_info(server_info(spec, options), policy, options)
    logger.info("server_info: %s", json.dumps(info))
    provider = build_tactile_provider(spec, options, info["tactile"]["fingertips"]) if live else None
    from gr00t.policy.server_client import PolicyServer

    server = PolicyServer(policy=policy, host=options.host, port=options.port)
    server.register_endpoint("server_info", lambda: info, requires_input=False)
    get_action = policy.get_action
    if provider is not None:
        get_action = with_live_tactile(policy, provider, settings["tactile_refresh"])
    if settings["cascade"]:
        get_action = with_cascade(policy, get_action, settings["cascade_fast_loops"])
    if get_action is not policy.get_action:
        server.register_endpoint("get_action", get_action)
    logger.info("serving %s on %s:%d", options.checkpoint, options.host, options.port)
    try:
        server.run()
    except KeyboardInterrupt:
        logger.info("stopped")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="gr00t-n17-tactile-server", description="Serve a tactile GR00T N1.7 checkpoint."
    )
    parser.add_argument("--task", default=DEFAULT_TASK, help="task id, e.g. gr00t_n17_tactile/h2_sharpa_tactile")
    parser.add_argument("--checkpoint", default="", help="finetuned checkpoint directory (no published default)")
    parser.add_argument("--host", default="0.0.0.0")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda", help="torch device the policy runs on")
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
        device=args.device,
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

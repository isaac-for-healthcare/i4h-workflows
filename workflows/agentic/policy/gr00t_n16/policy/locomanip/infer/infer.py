# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

from common.config import environment_config_path, get_policy_config
from common.health import PolicyHealth, serve_health

logger = logging.getLogger("policy")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-config-yaml-path", type=Path, default=None)
    parser.add_argument("--g1-model-repo", type=str, default=None)
    parser.add_argument("--num-envs", type=int, default=1)
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--execution-steps", type=int, default=None)
    parser.add_argument("--warmup-timeout", type=float, default=0.0)
    parser.add_argument("--health-host", type=str, default="0.0.0.0")
    parser.add_argument("--health-port", type=int, default=None)


def run(args: argparse.Namespace) -> None:
    from policy.locomanip.infer.closedloop_policy import G1LocomanipClosedloopPolicy
    from policy.locomanip.infer.io import PolicyIO

    policy_config = get_policy_config(args.env)
    args.health_port = args.health_port or policy_config.required_health_port
    policy_config_data = policy_config.to_dict()
    # Use a cached TRT engine unless YAML already specifies one.
    if not policy_config_data.get("trt_engine_path"):
        _default_engine = Path(__file__).resolve().parents[3] / "trt_engines" / args.env / "engines" / "dit_bf16.engine"
        if _default_engine.exists():
            policy_config_data["trt_engine_path"] = str(_default_engine)
            logger.info("Auto-detected TRT engine: %s", _default_engine)

    health = PolicyHealth()
    health_server = serve_health(health, host=args.health_host, port=args.health_port)
    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        logger.info("signal %s received, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    health.set("loading")
    model_repo = args.g1_model_repo or args.model_repo or policy_config.model_repo
    model_revision = args.model_revision or policy_config.model_revision
    logger.info(
        "Loading G1 GR00T policy: env=%s model=%s revision=%s",
        args.env,
        args.model_path or model_repo,
        model_revision or "<default>",
    )
    t0 = time.monotonic()
    policy = G1LocomanipClosedloopPolicy(
        args.policy_config_yaml_path,
        num_envs=args.num_envs,
        device=args.policy_device,
        model_path_override=args.model_path,
        model_repo=model_repo,
        model_revision=model_revision,
        policy_config_data=policy_config_data,
        config_base_dir=environment_config_path().parent.parent,
    )
    health.set_metadata(
        env=args.env,
        model_path=str(policy.resolved_model_path),
        model_repo=model_repo,
        model_revision=model_revision,
    )
    logger.info("G1 GR00T policy ready in %.1fs", time.monotonic() - t0)
    control_hz = args.control_hz or policy_config.control_hz
    execution_steps = args.execution_steps or policy_config.execution_steps
    if control_hz is None or execution_steps is None:
        raise ValueError(f"policy config for {args.env} must define control_hz and execution_steps")
    period = 1.0 / control_hz
    try:
        with PolicyIO(device=args.policy_device, num_envs=args.num_envs, env_id=args.env) as io:
            health.set("waiting_for_samples")
            logger.info("policy IO ready; waiting for G1 Arena samples")
            deadline = time.monotonic() + args.warmup_timeout if args.warmup_timeout > 0 else None
            while not stop:
                if io.wait_for_data(timeout=5.0):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for G1 Arena samples.")
            if stop:
                return

            health.set("running")
            last_consumed_ts = -1
            current_run_id = ""
            while not stop:
                obs = io.latest_observation()
                if obs is None or obs["state_ts"] == last_consumed_ts:
                    time.sleep(0.005)
                    continue
                obs_run_id = str(obs.get("run_id") or "unknown")
                if obs_run_id != current_run_id:
                    current_run_id = obs_run_id
                    policy.reset()
                    logger.info(
                        "policy run started: run_id=%s episode=%s attempt=%s state_ts=%s; reset action chunk state",
                        current_run_id,
                        obs.get("episode_index", 0),
                        obs.get("attempt_index", 0),
                        obs["state_ts"],
                    )
                t_inf = time.time_ns()
                action = policy.get_action(obs)
                io.publish_command(
                    action[:execution_steps],
                    dt=period,
                    inference_ts=t_inf,
                    run_id=obs.get("run_id"),
                    episode_index=obs.get("episode_index"),
                    attempt_index=obs.get("attempt_index"),
                )
                last_consumed_ts = obs["state_ts"]
    finally:
        health.set("stopping")
        health_server.shutdown()

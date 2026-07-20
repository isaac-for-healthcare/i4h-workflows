# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import signal
import time
from pathlib import Path

import numpy as np
from common.config import get_policy_config
from common.health import PolicyHealth, serve_health

logger = logging.getLogger("policy")


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--policy-device", default="cuda")
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--execution-steps", type=int, default=None)
    parser.add_argument("--warmup-timeout", type=float, default=0.0)
    parser.add_argument("--health-host", type=str, default="0.0.0.0")
    parser.add_argument("--health-port", type=int, default=None)
    parser.add_argument("--assemble-health-port", type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log one-line Assemble Trocar inference summaries every N inferences. Use 0 to disable.",
    )
    parser.add_argument(
        "--skip-assemble-rl-patch",
        action="store_true",
        help="Deprecated no-op; Assemble Trocar applies RL compatibility in-process.",
    )


def run(args: argparse.Namespace) -> None:
    from policy.assemble_trocar.infer.io import PolicyIO
    from policy.assemble_trocar.infer.policy import AssembleTrocarPolicy

    policy_config = get_policy_config(args.env)
    health_port = args.health_port or args.assemble_health_port or policy_config.required_health_port
    health = PolicyHealth()
    health_server = serve_health(health, host=args.health_host, port=health_port)
    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        logger.info("signal %s received, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    control_hz = args.control_hz or policy_config.control_hz
    execution_steps = args.execution_steps or policy_config.execution_steps
    if control_hz is None or execution_steps is None:
        raise ValueError(f"policy config for {args.env} must define control_hz and execution_steps")

    try:
        health.set("loading")
        t0 = time.monotonic()
        model_repo = args.model_repo or policy_config.required_model_repo
        model_revision = args.model_revision or policy_config.model_revision
        policy_config_data = policy_config.to_dict()
        trt_engine_path = policy_config_data.get("trt_engine_path")
        if not trt_engine_path:
            default_engine = (
                Path(__file__).resolve().parents[3] / "trt_engines" / args.env / "engines" / "dit_bf16.engine"
            )
            if default_engine.exists():
                trt_engine_path = str(default_engine)
                logger.info("Auto-detected TRT engine: %s", default_engine)
        policy = AssembleTrocarPolicy(
            model_path=args.model_path,
            model_repo=model_repo,
            model_revision=model_revision,
            task_description=policy_config.required_language_instruction,
            device=args.policy_device,
            action_head_future_tokens=policy_config.action_head_future_tokens,
            trt_engine_path=trt_engine_path,
        )
        health.set_metadata(
            env=args.env,
            model_path=str(policy.model_path),
            model_repo=model_repo,
            model_revision=model_revision,
        )
        logger.info("Assemble Trocar policy ready in %.1fs", time.monotonic() - t0)
        period = 1.0 / control_hz
        with PolicyIO(env_id=args.env) as io:
            health.set("waiting_for_samples")
            logger.info("policy IO ready; waiting for Assemble Trocar Arena samples")
            deadline = time.monotonic() + args.warmup_timeout if args.warmup_timeout > 0 else None
            while not stop:
                if io.wait_for_data(timeout=5.0):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for Assemble Trocar Arena samples.")
            if stop:
                return

            health.set("running")
            last_consumed_ts = -1
            current_run_id = ""
            inference_count = 0
            while not stop:
                obs = io.latest_observation()
                if obs is None or obs["state_ts"] == last_consumed_ts:
                    time.sleep(0.005)
                    continue
                obs_run_id = str(obs.get("run_id") or "unknown")
                if obs_run_id != current_run_id:
                    current_run_id = obs_run_id
                    logger.info(
                        "policy run started: run_id=%s episode=%s attempt=%s state_ts=%s",
                        current_run_id,
                        obs.get("episode_index", 0),
                        obs.get("attempt_index", 0),
                        obs["state_ts"],
                    )
                t_inf = time.time_ns()
                t0 = time.monotonic()
                action = policy.get_action(obs)
                latency_ms = (time.monotonic() - t0) * 1000.0
                io.publish_command(
                    action[:execution_steps],
                    dt=period,
                    inference_ts=t_inf,
                    run_id=obs.get("run_id"),
                    episode_index=obs.get("episode_index"),
                    attempt_index=obs.get("attempt_index"),
                )
                inference_count += 1
                _log_inference(args, inference_count, latency_ms, obs, action[:execution_steps])
                last_consumed_ts = obs["state_ts"]
    finally:
        health.set("stopping")
        health_server.shutdown()


def _log_inference(
    args: argparse.Namespace,
    inference_count: int,
    latency_ms: float,
    obs: dict,
    executed,
) -> None:
    if args.log_every <= 0:
        return
    if inference_count != 1 and inference_count % args.log_every != 0:
        return

    action = np.asarray(executed, dtype=np.float32)
    joints = np.asarray(obs["joint_positions"], dtype=np.float32)
    frames = obs.get("frames", {})
    frame_shapes = {key: tuple(value.shape) for key, value in frames.items() if value is not None}
    logger.info(
        "Assemble Trocar inference=%s infer=%.1fms actions=%s range=(%.3f, %.3f) state_dim=%s cameras=%s",
        inference_count,
        latency_ms,
        tuple(action.shape),
        float(action.min()) if action.size else 0.0,
        float(action.max()) if action.size else 0.0,
        joints.shape[-1] if joints.ndim else 0,
        frame_shapes,
    )

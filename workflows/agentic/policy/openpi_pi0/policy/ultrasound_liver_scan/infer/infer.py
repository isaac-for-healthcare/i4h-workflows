# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import argparse
import logging
import os
import signal
import time
from pathlib import Path

from common.config import get_policy_config
from common.health import PolicyHealth, serve_health

logger = logging.getLogger("policy")

# Prevent JAX from preallocating all GPU memory.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--repo-id", type=str, default="i4h/sim_liver_scan", help="LeRobot repo id for dataset normalization stats."
    )
    p.add_argument(
        "--task", type=str, default=None, help="Task description. Defaults to the env's language_instruction."
    )
    p.add_argument("--control-hz", type=float, default=30.0)
    p.add_argument(
        "--action-horizon", type=int, default=50, help="Chunk length the PI0 model emits per inference (default 50)."
    )
    p.add_argument(
        "--execution-steps", type=int, default=50, help="Action steps to publish per inference (<= action-horizon)."
    )
    p.add_argument("--warmup-timeout", type=float, default=0.0)
    p.add_argument("--health-host", type=str, default="0.0.0.0")
    p.add_argument("--health-port", type=int, default=None)
    p.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log one-line inference summaries every N inferences. Use 0 to disable.",
    )


def run(args: argparse.Namespace) -> None:
    from policy.ultrasound_liver_scan.infer.io import PolicyIO
    from policy.ultrasound_liver_scan.infer.runner import PI0PolicyRunner, RunnerConfig

    _validate_args(args)
    policy_config = get_policy_config(args.env)
    args.health_port = args.health_port or policy_config.required_health_port
    task_description = args.task or policy_config.required_language_instruction
    model_path = _resolve_model_path(
        args.model_path,
        args.model_repo or policy_config.required_model_repo,
    )
    model_repo = args.model_repo or policy_config.required_model_repo

    health = PolicyHealth()
    health_server = serve_health(health, host=args.health_host, port=args.health_port)
    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        logger.info("signal %s received, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    runner = PI0PolicyRunner(
        RunnerConfig(
            model_path=model_path,
            repo_id=args.repo_id,
            task_description=task_description,
        )
    )
    health.set_metadata(
        env=args.env,
        model_path=str(Path(model_path).expanduser().resolve()),
        model_repo=model_repo,
        model_revision=getattr(args, "model_revision", None),
    )
    health.set("loading")
    runner.ensure_loaded()

    period = 1.0 / args.control_hz
    try:
        with PolicyIO(env_id=args.env) as io:
            health.set("waiting_for_samples")
            logger.info("policy IO ready; waiting for ultrasound Arena samples")
            deadline = time.monotonic() + args.warmup_timeout if args.warmup_timeout > 0 else None
            while not stop:
                if io.wait_for_data(timeout=5.0):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for ultrasound Arena samples.")
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
                actions = runner.infer(obs, num_steps=args.action_horizon)
                latency_ms = (time.monotonic() - t0) * 1000.0
                io.publish_command(
                    actions[: args.execution_steps],
                    dt=period,
                    inference_ts=t_inf,
                    run_id=obs.get("run_id"),
                    episode_index=obs.get("episode_index"),
                    attempt_index=obs.get("attempt_index"),
                )
                inference_count += 1
                _log_inference(args, inference_count, latency_ms, actions[: args.execution_steps])
                last_consumed_ts = obs["state_ts"]
    finally:
        health.set("stopping")
        health_server.shutdown()


def _resolve_model_path(model_path: str | None, model_repo: str) -> str:
    if model_path:
        return model_path
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("ULTRASOUND_MODEL_CACHE", os.path.expanduser("~/.cache/ultrasound_models"))
    os.makedirs(cache_dir, exist_ok=True)
    logger.info("No --model-path given; pulling %s into %s", model_repo, cache_dir)
    return snapshot_download(repo_id=model_repo, cache_dir=cache_dir)


def _validate_args(args: argparse.Namespace) -> None:
    if args.action_horizon < 1:
        raise SystemExit("--action-horizon must be >= 1")
    if args.execution_steps < 1:
        raise SystemExit("--execution-steps must be >= 1")
    if args.execution_steps > args.action_horizon:
        raise SystemExit("--execution-steps must be <= --action-horizon")


def _log_inference(args: argparse.Namespace, inference_count: int, latency_ms: float, executed) -> None:
    if args.log_every <= 0:
        return
    if inference_count != 1 and inference_count % args.log_every != 0:
        return
    abs_executed = abs(executed)
    logger.info(
        "Ultrasound PI0 inference=%s infer=%.1fms actions=%s range=(%.3f, %.3f) " "abs_mean=%.4f max_abs=%.4f first=%s",
        inference_count,
        latency_ms,
        tuple(executed.shape),
        float(executed.min()) if executed.size else 0.0,
        float(executed.max()) if executed.size else 0.0,
        float(abs_executed.mean()) if executed.size else 0.0,
        float(abs_executed.max()) if executed.size else 0.0,
        _format_action_row(executed[0]) if executed.size else "[]",
    )


def _format_action_row(row) -> str:
    return "[" + ", ".join(f"{float(value):.3f}" for value in row.tolist()) + "]"

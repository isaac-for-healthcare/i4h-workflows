# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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
SO_ARM_POLICY_CONFIG = get_policy_config("scissor_pick_and_place")

# Default location for engines built by scripts/build_trt_engines.sh.
# infer.py -> scissor_pick_and_place -> policy (pkg) -> gr00t_n17 (project).
# Engines live env-scoped: ``trt_engines/<env_id>/engines/``, matching the
# gr00t_n15 layout where multiple envs each have their own engine set.
_DEFAULT_TRT_ENGINE_DIR = Path(__file__).resolve().parents[2] / "trt_engines" / "scissor_pick_and_place" / "engines"


def _default_trt_engine_path() -> str | None:
    """Return the local engines dir if the build artifacts exist, else None.

    Lets users build engines once with ``scripts/build_trt_engines.sh`` and
    have the daemon pick them up automatically — no per-run CLI flag needed.
    Explicit ``--trt-engine-path`` and env config settings still win.
    """
    if _DEFAULT_TRT_ENGINE_DIR.is_dir() and any(_DEFAULT_TRT_ENGINE_DIR.glob("*.engine")):
        return str(_DEFAULT_TRT_ENGINE_DIR)
    return None


def add_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--model-path", type=str, default=None)
    p.add_argument("--model-repo", type=str, default=None)
    p.add_argument("--task", type=str, default=SO_ARM_POLICY_CONFIG.task_description)
    p.add_argument("--embodiment-tag", type=str, default=SO_ARM_POLICY_CONFIG.embodiment_tag)
    p.add_argument(
        "--trt-engine-path",
        type=str,
        default=SO_ARM_POLICY_CONFIG.trt_engine_path or _default_trt_engine_path(),
        help=(
            "Directory containing pre-built TensorRT engines. Activates TRT inference. "
            "Auto-detected from policy/trt_engines/engines/ when present."
        ),
    )
    p.add_argument(
        "--trt-mode",
        type=str,
        default="n17_full_pipeline",
        choices=["n17_full_pipeline", "vit_llm_only", "action_head", "dit_only"],
    )
    p.add_argument("--control-hz", type=float, default=SO_ARM_POLICY_CONFIG.control_hz or 30.0)
    p.add_argument("--action-horizon", type=int, default=SO_ARM_POLICY_CONFIG.action_horizon)
    p.add_argument("--execution-steps", type=int, default=SO_ARM_POLICY_CONFIG.execution_steps)
    p.add_argument("--warmup-timeout", type=float, default=0.0)
    p.add_argument("--lazy-load", action="store_true")
    p.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log one-line inference summaries every N inferences. Use 0 to disable.",
    )
    p.add_argument(
        "--log-actions",
        action="store_true",
        help="Log detailed state/action vectors for early inferences.",
    )
    p.add_argument("--health-host", type=str, default=os.environ.get("SOARM_POLICY_HEALTH_HOST", "0.0.0.0"))
    p.add_argument("--health-port", type=int, default=None)


def run(args: argparse.Namespace) -> None:
    from policy.scissor_pick_and_place.infer.io import PolicyIO
    from policy.scissor_pick_and_place.infer.runner import GR00TPolicyRunner, RunnerConfig

    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    _validate_args(args)
    args.health_port = args.health_port or SO_ARM_POLICY_CONFIG.required_health_port

    health = PolicyHealth()
    health_server = serve_health(health, host=args.health_host, port=args.health_port)
    logger.info("policy daemon started (health=http://%s:%s/healthz)", args.health_host, args.health_port)

    model_repo = args.model_repo or SO_ARM_POLICY_CONFIG.required_model_repo
    model_path = _resolve_model_path(args.model_path, model_repo)
    health.set_metadata(
        env=args.env,
        model_path=str(Path(model_path).expanduser().resolve()),
        model_repo=model_repo,
        model_revision=getattr(args, "model_revision", None),
    )
    runner = GR00TPolicyRunner(
        RunnerConfig(
            model_path=model_path,
            task_description=args.task,
            embodiment_tag=args.embodiment_tag,
            trt_engine_path=args.trt_engine_path,
            trt_mode=args.trt_mode,
        )
    )

    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        logger.info("signal %s received, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    if not args.lazy_load:
        health.set("loading")
        runner.ensure_loaded()

    period = 1.0 / args.control_hz

    try:
        with PolicyIO(env_id=args.env) as io:
            health.set("waiting_for_samples")
            logger.info("policy IO ready; waiting for cameras and state")
            deadline = time.monotonic() + args.warmup_timeout if args.warmup_timeout > 0 else None
            while not stop:
                if io.wait_for_data(timeout=5.0):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for cameras and state. Is Arena running?")
            if stop:
                return

            health.set("running")
            logger.info(
                "policy connected; publishing %s/%s action steps at %.1fHz; log_every=%s",
                args.execution_steps,
                args.action_horizon,
                args.control_hz,
                args.log_every,
            )
            last_consumed_ts = -1
            total_inference_count = 0
            run_id = ""
            run_inference_count = 0
            while not stop:
                obs = io.latest_observation()
                if obs is None or obs["state_ts"] == last_consumed_ts:
                    time.sleep(0.005)
                    continue
                obs_run_id = str(obs.get("run_id") or "unknown")
                if obs_run_id != run_id:
                    run_id = obs_run_id
                    run_inference_count = 0
                    logger.info(
                        "policy run started: run_id=%s episode=%s attempt=%s state_ts=%s",
                        run_id,
                        obs.get("episode_index", 0),
                        obs.get("attempt_index", 0),
                        obs["state_ts"],
                    )
                t_inf = time.time_ns()
                t0 = time.monotonic()
                chunk = runner.infer(obs, num_steps=args.action_horizon)
                latency_ms = (time.monotonic() - t0) * 1000.0
                io.publish_command(
                    chunk[: args.execution_steps],
                    dt=period,
                    inference_ts=t_inf,
                    run_id=obs.get("run_id"),
                    episode_index=obs.get("episode_index"),
                    attempt_index=obs.get("attempt_index"),
                )
                total_inference_count += 1
                run_inference_count += 1
                _log_inference(
                    args,
                    run_id,
                    run_inference_count,
                    total_inference_count,
                    latency_ms,
                    obs,
                    chunk[: args.execution_steps],
                )
                last_consumed_ts = obs["state_ts"]
    finally:
        health.set("stopping")
        health_server.shutdown()


def _resolve_model_path(model_path: str | None, model_repo: str) -> str:
    if model_path:
        return model_path
    from huggingface_hub import snapshot_download

    cache_dir = os.environ.get("SOARM_MODEL_CACHE", os.path.expanduser("~/.cache/soarm_models"))
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
    if args.log_every < 0:
        raise SystemExit("--log-every must be >= 0")


def _log_inference(
    args: argparse.Namespace,
    run_id: str,
    run_inference_count: int,
    total_inference_count: int,
    latency_ms: float,
    obs: dict,
    executed,
) -> None:
    should_log_summary = args.log_every > 0 and (run_inference_count == 1 or run_inference_count % args.log_every == 0)
    should_log_detail = args.log_actions and run_inference_count <= 5
    if not should_log_summary and not should_log_detail:
        return

    action_min = float(executed.min()) if executed.size else 0.0
    action_max = float(executed.max()) if executed.size else 0.0
    if should_log_detail:
        logger.info(
            "run=%s inference=%s detail: total=%s infer=%.1fms state=%s first_action=%s range=(%.2f, %.2f)",
            run_id,
            run_inference_count,
            total_inference_count,
            latency_ms,
            _fmt_values(obs["joint_positions"]),
            _fmt_values(executed[0]) if executed.size else "[]",
            action_min,
            action_max,
        )
        return

    logger.info(
        "run=%s inference=%s: infer=%.1fms published=%s steps grip(state/first/last)=%.2f/%.2f/%.2f action_range=(%.2f, %.2f) total=%s",
        run_id,
        run_inference_count,
        latency_ms,
        int(executed.shape[0]) if executed.ndim >= 1 else 0,
        _last_scalar(obs["joint_positions"]),
        _action_gripper(executed, 0),
        _action_gripper(executed, -1),
        action_min,
        action_max,
        total_inference_count,
    )


def _fmt_values(values, *, limit: int = 6) -> str:
    flat = list(values[:limit])
    suffix = "" if len(values) <= limit else ", ..."
    return "[" + ", ".join(f"{float(value):.3f}" for value in flat) + suffix + "]"


def _last_scalar(values) -> float:
    return float(values[-1]) if len(values) else 0.0


def _action_gripper(actions, index: int) -> float:
    if not actions.size:
        return 0.0
    return float(actions[index, -1] if actions.ndim >= 2 else actions[-1])

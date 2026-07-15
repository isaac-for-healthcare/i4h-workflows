# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Generic surgical baseline policy daemon.

This is an inference-path smoke policy, not a trained surgical manipulation
model. It subscribes to Arena camera/state samples and publishes zero action
chunks so the normal policy daemon, health, Zenoh, and Arena command loop can
be validated for new surgical environments.
"""

from __future__ import annotations

import argparse
import logging
import signal
import time
from typing import Optional

import numpy as np
from common.config import get_env_robot_config, get_policy_config
from common.health import PolicyHealth, serve_health
from common.io.policy import PolicyIOBase, camera_to_array

logger = logging.getLogger("policy")


class PolicyIO(PolicyIOBase):
    def latest_observation(self) -> Optional[dict]:
        with self._lock:
            if self._state is None or not self._camera_names.issubset(self._frames):
                return None
            frames = {}
            for cam_key, frame in self._frames.items():
                image = camera_to_array(frame)
                if image is None:
                    return None
                frames[cam_key] = image
            obs = {
                "frames": frames,
                "joint_positions": np.asarray(self._state.joint_positions, dtype=np.float64).copy(),
                "state_ts": self._state.ts,
                "run_id": self._state.run_id,
                "episode_index": self._state.episode_index,
                "attempt_index": self._state.attempt_index,
            }
        return obs


def add_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--control-hz", type=float, default=None)
    parser.add_argument("--action-horizon", type=int, default=None)
    parser.add_argument("--execution-steps", type=int, default=None)
    parser.add_argument("--warmup-timeout", type=float, default=0.0)
    parser.add_argument("--health-host", type=str, default="0.0.0.0")
    parser.add_argument("--health-port", type=int, default=None)
    parser.add_argument(
        "--log-every",
        type=int,
        default=0,
        help="Log one-line baseline inference summaries every N inferences. Use 0 to disable.",
    )


def run(args: argparse.Namespace) -> None:
    policy_config = get_policy_config(args.env)
    robot_config = get_env_robot_config(args.env)
    health_port = args.health_port or policy_config.required_health_port
    action_dim = robot_config.action_dim
    action_horizon = args.action_horizon or policy_config.action_horizon or 16
    execution_steps = args.execution_steps or policy_config.execution_steps or action_horizon
    control_hz = args.control_hz or policy_config.control_hz or 60.0
    if execution_steps < 1 or action_horizon < 1:
        raise ValueError("action_horizon and execution_steps must be positive")
    if execution_steps > action_horizon:
        raise ValueError("execution_steps must be <= action_horizon")

    health = PolicyHealth()
    health_server = serve_health(health, host=args.health_host, port=health_port)
    stop = False

    def _on_signal(signum, _frame):
        nonlocal stop
        logger.info("signal %s received, shutting down", signum)
        stop = True

    signal.signal(signal.SIGINT, _on_signal)
    signal.signal(signal.SIGTERM, _on_signal)

    try:
        health.set_metadata(
            env=args.env,
            model_repo="baseline-zero-action",
            model_revision="smoke",
            action_dim=action_dim,
        )
        period = 1.0 / control_hz
        logger.info(
            "surgical baseline policy ready: env=%s action_dim=%s horizon=%s execution_steps=%s control_hz=%.1f",
            args.env,
            action_dim,
            action_horizon,
            execution_steps,
            control_hz,
        )
        with PolicyIO(env_id=args.env) as io:
            health.set("waiting_for_samples")
            logger.info("policy IO ready; waiting for surgical Arena samples")
            deadline = time.monotonic() + args.warmup_timeout if args.warmup_timeout > 0 else None
            while not stop:
                if io.wait_for_data(timeout=5.0):
                    break
                if deadline is not None and time.monotonic() >= deadline:
                    raise SystemExit("Timed out waiting for surgical Arena samples.")
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
                        "baseline policy run started: run_id=%s episode=%s attempt=%s state_ts=%s",
                        current_run_id,
                        obs.get("episode_index", 0),
                        obs.get("attempt_index", 0),
                        obs["state_ts"],
                    )
                action = np.zeros((action_horizon, action_dim), dtype=np.float64)
                io.publish_command(
                    action[:execution_steps],
                    dt=period,
                    inference_ts=time.time_ns(),
                    run_id=obs.get("run_id"),
                    episode_index=obs.get("episode_index"),
                    attempt_index=obs.get("attempt_index"),
                )
                inference_count += 1
                _log_inference(args, inference_count, obs, action[:execution_steps])
                last_consumed_ts = obs["state_ts"]
    finally:
        health.set("stopping")
        health_server.shutdown()


def _log_inference(args: argparse.Namespace, inference_count: int, obs: dict, executed: np.ndarray) -> None:
    if args.log_every <= 0:
        return
    if inference_count != 1 and inference_count % args.log_every != 0:
        return
    joints = np.asarray(obs["joint_positions"], dtype=np.float64)
    frame_shapes = {key: tuple(value.shape) for key, value in obs.get("frames", {}).items()}
    logger.info(
        "surgical baseline inference=%s actions=%s state_dim=%s cameras=%s",
        inference_count,
        tuple(executed.shape),
        joints.shape[-1] if joints.ndim else 0,
        frame_shapes,
    )

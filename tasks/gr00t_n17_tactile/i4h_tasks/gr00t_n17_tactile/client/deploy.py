# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive a Unitree H2 + SharpaWave hands from a ``gr00t_n17_tactile`` policy server.

Two processes, one workstation: the policy server (``i4h-tasks-gr00t-n17-tactile``,
task venv, GPU) and this client (a Python 3.10 environment with ROS 2 and the
Unitree DDS SDK, see the README). The client asks the server what the served
checkpoint expects (``server_info``), builds one observation per request from
the cameras, the joint state and the tactile stream, and walks the returned
action chunks at the control rate:

* **sync**: observe, infer, execute ``execution_steps`` targets, repeat;
* **async** (``--async-policy``): a worker thread keeps inferring while the
  main thread executes the newest chunk from the step its capture time maps to
  (real-time chunking), optionally refreshing the unexecuted tail from fresher
  chunks (``--trex-action-execution``);
* **same-plan** (``--async-policy --trex-refine-mode same-plan``): the client
  drives the server's slow/fast cascade explicitly, refining the tail of the
  *same* plan with fresh tactile data between chunk boundaries.

Episodes are framed by a go-home hold and an optional recorded init posture,
with a robot-side tactile tare before each rollout and an operator label per
episode when ``--log-results`` is set. ``--check`` builds one observation and
writes the frames to disk without contacting the server; ``--fake-robot``
substitutes synthetic cameras and joints so the whole loop runs against a
local policy server without hardware.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import yaml

from i4h_tasks.gr00t_n17_tactile.client.execution import (
    AsyncPolicyWorker,
    ChunkExecutor,
    SmoothingConfig,
    TRexPlanWorker,
    apply_new_plan_refresh,
    apply_same_plan_refine,
    chunk_length,
    copy_chunk,
    pace,
    plan_horizon,
    refine_due,
    refresh_due,
    slow_due,
    start_step_for,
)
from i4h_tasks.gr00t_n17_tactile.client.observation import (
    DEFAULT_FORCE_WINDOW,
    EYE_KEYS,
    CameraLayout,
    ObservationSpec,
    SpecOverrides,
    build_observation,
    describe_observation,
    read_contact_metric,
    read_tactile,
    resolve_layout,
    resolve_spec,
)
from i4h_tasks.gr00t_n17_tactile.client.protocol import GR00TClient, describe_server_info, wait_for_server
from i4h_tasks.gr00t_n17_tactile.client.robot import (
    ArmOptions,
    CameraOptions,
    DdsOptions,
    FakeRobot,
    H2Robot,
    RobotIO,
    RobotOptions,
    home_and_hold,
    init_and_hold,
)
from i4h_tasks.gr00t_n17_tactile.tactile.interface import HANDS

logger = logging.getLogger("i4h_tasks.gr00t_n17_tactile.client")

PROG = "i4h-tasks-gr00t-n17-tactile-deploy"
DEFAULT_CONTROL_HZ = 30.0
DEFAULT_EXECUTION_STEPS = 16
#: How long a chunk acquisition may block before the run is declared stuck.
CHUNK_WAIT_S = 30.0
#: Seconds between repeats of a per-tick warning (missing frames or fingertips).
WARN_EVERY_S = 5.0
LABEL_SHORTCUTS = {"s": "success", "f": "fail", "a": "abort", "r": "retry"}


# ── command line ─────────────────────────────────────────────────────────────


def build_parser() -> argparse.ArgumentParser:
    """Every flag of the old client that the pinned third-party set supports, with its name kept."""
    ap = argparse.ArgumentParser(
        prog=PROG, description="Drive a Unitree H2 + SharpaWave hands from a gr00t_n17_tactile policy server."
    )
    bool_flag = argparse.BooleanOptionalAction

    g = ap.add_argument_group("server")
    g.add_argument("--policy-host", default="localhost", help="policy server host")
    g.add_argument("--policy-port", type=int, default=5555, help="policy server port")
    g.add_argument("--timeout-ms", type=int, default=15000, help="ZMQ receive/send timeout")
    g.add_argument("--task", default="", help="language instruction sent with every observation")

    g = ap.add_argument_group("run")
    g.add_argument(
        "--check", action="store_true", help="build and print one observation, save frames, send nothing to the server"
    )
    g.add_argument("--check-dir", default="/tmp", help="where --check writes check_<key>.png / check_tactile_<key>.png")
    g.add_argument(
        "--fake-robot",
        action="store_true",
        help="no hardware; synthetic cameras and state, commands recorded and logged (smokes against a local server)",
    )
    g.add_argument(
        "--frequency", type=float, default=None, help="control rate in Hz; default: server control_hz, else 30"
    )
    g.add_argument(
        "--action-horizon",
        type=int,
        default=None,
        help="steps executed per chunk before re-querying; default: server execution_steps, else 16",
    )
    g.add_argument("--steps", type=int, default=30, help="control steps per episode")
    g.add_argument(
        "--num-episodes", type=int, default=None, help="stop after this many labelled episodes; default unlimited"
    )
    g.add_argument(
        "--log-results",
        default=None,
        help="append one JSONL record per labelled episode here (operator prompt s/f/a/r)",
    )
    g.add_argument("--hold-s", type=int, default=20, help="seconds to hold at home before / between / after episodes")
    g.add_argument(
        "--init-state-yaml",
        default=None,
        help="compute_average_initial_state.py output (states.left_arm.qpos, ...) staged after go-home",
    )
    g.add_argument("--init-state-hold-s", type=int, default=3, help="seconds to hold the init posture")
    g.add_argument(
        "--right-arm-hold-qpos",
        type=float,
        nargs=7,
        default=None,
        metavar="Q",
        help="fixed right-arm pose for a single-arm (left) deployment; overrides the YAML; "
        "default: the measured post-home pose",
    )

    g = ap.add_argument_group("real-time chunking")
    g.add_argument(
        "--async-policy", action="store_true", help="run inference concurrently with execution (real-time chunking)"
    )
    g.add_argument(
        "--async-cache-poll-s", type=float, default=0.005, help="poll interval for a fresh cached chunk between chunks"
    )
    g.add_argument(
        "--smooth-alpha", type=float, default=1.0, help="EMA on per-step targets, alpha*raw + (1-alpha)*prev; 1.0 = off"
    )
    g.add_argument(
        "--boundary-smooth-alpha",
        type=float,
        default=1.0,
        help="EMA alpha during the first --boundary-smooth-steps ticks after a chunk switch",
    )
    g.add_argument(
        "--boundary-smooth-steps",
        type=int,
        default=0,
        help="ticks after a chunk switch that use --boundary-smooth-alpha; 0 disables",
    )
    g.add_argument(
        "--boundary-blend-steps",
        type=int,
        default=0,
        help="cross-chunk blend length: fade the old chunk's unexecuted future into the new chunk; 0 disables",
    )
    g.add_argument(
        "--boundary-blend-alpha",
        type=float,
        default=2.0,
        help="exponential decay of the blend, w_old(t) = exp(-alpha * t / N); larger = faster fade-out",
    )
    g.add_argument(
        "--trex-action-execution",
        action="store_true",
        help="async only: refresh the unexecuted tail in place from the freshest worker chunk (new-plan mode)",
    )
    g.add_argument(
        "--trex-refine-every",
        type=int,
        default=4,
        help="refresh cadence: new-plan every N march offsets; same-plan on plan index t when t %% N == 0",
    )
    g.add_argument(
        "--trex-refine-mode",
        choices=["new-plan", "same-plan"],
        default="new-plan",
        help="new-plan: splice the freshest independent chunk; same-plan: refine the same cached plan "
        "(requires --async-policy and a server with cascade)",
    )
    g.add_argument(
        "--trex-max-refine-offset",
        type=int,
        default=4,
        help="same-plan only: refine only at plan index t <= this (the checkpoint's delay-aug coverage)",
    )
    g.add_argument(
        "--trex-slow-lead",
        type=int,
        default=3,
        help="same-plan only: submit the next plan's slow pass this many ticks before the chunk boundary",
    )
    g.add_argument(
        "--trex-contact-gate",
        action="store_true",
        help="same-plan only: refine only when the tactile force magnitude >= --trex-contact-threshold",
    )
    g.add_argument(
        "--trex-contact-threshold",
        type=float,
        default=4.0,
        help="contact metric (max |f6| over the left fingers) at which --trex-contact-gate lets a refine through",
    )
    g.add_argument(
        "--trex-refine-blend",
        action="store_true",
        help="same-plan only: EMA-blend each refine into the tail instead of overwriting it",
    )
    g.add_argument(
        "--trex-refine-blend-alpha", type=float, default=0.5, help="weight on the newest refine for --trex-refine-blend"
    )

    g = ap.add_argument_group("robot")
    g.add_argument(
        "--arm", action=bool_flag, default=True, help="drive the H2 arms (off = zero arm state, no arm commands)"
    )
    g.add_argument(
        "--hands", action=bool_flag, default=True, help="drive the SharpaWave hands through the robot-side DDS bridge"
    )
    g.add_argument("--motion", action="store_true", help="publish on rt/arm_sdk (standing robot); default rt/lowcmd")
    g.add_argument(
        "--head-pitch-home",
        type=float,
        default=0.6,
        metavar="RAD",
        help="head pitch held during inference, radians (-0.523 up .. 0.837 down)",
    )
    g.add_argument(
        "--arm-velocity-limit", type=float, default=20.0, metavar="RAD_S", help="flat arm joint speed cap, rad/s"
    )
    for name in ("kp-low", "kp-wrist", "kd-low", "kd-wrist", "kp-head", "kd-head"):
        g.add_argument(
            f"--{name}",
            type=float,
            default=None,
            help=f"{name.replace('-', ' ')} PD gain passed to H2_ArmController "
            "(controller defaults 150/50/10/3/150/10)",
        )
    g.add_argument("--dds-domain", type=int, default=0, help="Unitree DDS domain")
    g.add_argument(
        "--network-interface", default=None, help="Unitree DDS NIC name (bypasses the address pinning below)"
    )
    g.add_argument(
        "--network-address",
        default=None,
        help="workstation IP on the robot network (default teleop's WORKSTATION_ROBOT_DDS_ADDRESS)",
    )
    g.add_argument("--network-peer", default=None, help="onboard-computer DDS peer (default teleop's THOR_DDS_ADDRESS)")
    g.add_argument(
        "--body-dds-peer",
        action="append",
        default=None,
        metavar="IP[:PORT]",
        help="H2 body DDS peer, repeatable; default: discovered through the onboard computer",
    )
    g.add_argument(
        "--body-dds-discovery-ssh",
        default=None,
        metavar="[USER@]HOST",
        help="SSH target for the SPDP probe (default teleop's DEFAULT_THOR_SSH_TARGET)",
    )
    g.add_argument(
        "--body-dds-discovery-timeout", type=float, default=35.0, help="seconds to wait for both body participants"
    )

    g = ap.add_argument_group("cameras")
    g.add_argument(
        "--head-mode",
        choices=["binocular", "mono"],
        default="mono",
        help="/head/left alone (default; what the three-camera launcher publishes) or the stitched stereo head "
        "(/head/left + /head/right) for eye-view checkpoints",
    )
    g.add_argument(
        "--head-eye",
        choices=["left", "right", "both"],
        default=None,
        help="which eye is video.high; default: both when the server lists eye views, else left",
    )
    g.add_argument("--cam-warmup-s", type=float, default=5.0, help="wait for the first frame of each camera")
    g.add_argument(
        "--swap-wrist-cams",
        action=bool_flag,
        default=False,
        help="exchange the two wrist streams before they become left_wrist_view / right_wrist_view; off by default "
        "(identity routing reproduces the ROS-recorded training data), on for the legacy ZMQ-era checkpoints",
    )
    g.add_argument(
        "--camera-network-address",
        default=None,
        help="workstation IP on the camera network; with the peer, pins CycloneDDS for the ROS client",
    )
    g.add_argument("--camera-network-peer", default=None, help="onboard-computer IP on the camera network")
    g.add_argument(
        "--wrist-cams",
        choices=["auto", "right", "left", "both", "none"],
        default="auto",
        help="wrist-view keys to send; auto = the server's video keys",
    )

    g = ap.add_argument_group("observation / checkpoint")
    g.add_argument(
        "--state-hands",
        choices=["auto", "both", "left", "right"],
        default="auto",
        help="hand state keys to send (and fingertips to stream); auto = the server's state keys",
    )
    g.add_argument(
        "--checkpoint-tactile",
        action=bool_flag,
        default=None,
        help="force whether tactile keys are sent; default from server_info",
    )
    g.add_argument(
        "--checkpoint-tactile-force",
        action=bool_flag,
        default=None,
        help="force whether the force window is sent; default from server_info",
    )
    g.add_argument(
        "--tactile",
        action=bool_flag,
        default=True,
        help="subscribe to the live tactile stream on the client; off = all-zero maps and force (ablation arm)",
    )
    g.add_argument("--tactile-host", default="127.0.0.1", help="tactile bridge host")
    g.add_argument("--tactile-port", type=int, default=7779, help="tactile bridge port")

    g = ap.add_argument_group("tactile calibration (robot-side tare before each rollout)")
    g.add_argument(
        "--tactile-calib",
        action=bool_flag,
        default=None,
        help="tare both hands' tactile over SSH at the init pose and before every episode; "
        "default: auto (on iff tactile keys are sent and --tactile is on)",
    )
    g.add_argument(
        "--tactile-calib-ssh",
        default=None,
        help="SSH target of the tactile container (default unitree@<--tactile-host>)",
    )
    g.add_argument(
        "--tactile-calib-container", default="sharpa180", help="container holding calibrate_sharpa_tactile.py"
    )
    g.add_argument(
        "--tactile-calib-attempts", type=int, default=3, help="calibration attempts before prompting the operator"
    )
    g.add_argument(
        "--tactile-calib-settle-s", type=float, default=1.0, help="settle time after a successful tare, seconds"
    )

    ap.add_argument("--verbose", action="store_true", help="DEBUG logging")
    return ap


# ── settings resolved from server_info ───────────────────────────────────────


@dataclass
class RunSettings:
    """Everything resolved once after connecting; pure, so it is testable without hardware."""

    spec: ObservationSpec
    layout: CameraLayout
    smoothing: SmoothingConfig
    control_hz: float
    dt: float
    execution_steps: int
    steps: int
    tactile_live: bool
    tactile_calib: bool
    mode: str
    trex_action_execution: bool
    warnings: list[str] = field(default_factory=list)


def _section(info: dict[str, Any] | None, key: str) -> dict[str, Any]:
    value = (info or {}).get(key)
    return value if isinstance(value, dict) else {}


def resolve_run_settings(args: argparse.Namespace, info: dict[str, Any] | None) -> RunSettings:
    """What the run looks like given the flags and what the server said about its checkpoint.

    Explicit flags win over ``server_info`` with a warning; ``info`` ``None``
    (a server without the endpoint, or ``--check``) falls back to the legacy
    defaults. Raises ``ValueError`` for a force window without tactile keys and
    for same-plan against a server known to run without its cascade.
    """
    overrides = SpecOverrides(
        checkpoint_tactile=args.checkpoint_tactile,
        checkpoint_tactile_force=args.checkpoint_tactile_force,
        state_hands=args.state_hands,
        wrist_cams=args.wrist_cams,
        head_eye=args.head_eye,
    )
    spec, warnings = resolve_spec(info, overrides)
    warnings = list(warnings)
    layout = resolve_layout(spec, args.head_mode, args.head_eye, args.swap_wrist_cams)
    if layout.head_mode == "mono" and any(key in spec.video_keys for key in EYE_KEYS):
        warnings.append(
            "the checkpoint wants eye views but --head-mode mono reads /head/left only; pass --head-mode binocular "
            "(needs the /head/right camera)"
        )

    control_hz = float(args.frequency or (info or {}).get("control_hz") or DEFAULT_CONTROL_HZ)
    execution_steps = int(args.action_horizon or (info or {}).get("execution_steps") or DEFAULT_EXECUTION_STEPS)
    action_horizon = (info or {}).get("action_horizon")
    if action_horizon and execution_steps > int(action_horizon):
        warnings.append(
            f"execution_steps={execution_steps} exceeds the server's action_horizon={action_horizon}; "
            "each chunk is shorter than requested"
        )

    if args.tactile_calib is None:
        tactile_calib = bool(spec.uses_tactile and args.tactile)
        logger.info("tactile_calib=%s  [auto]", tactile_calib)
    else:
        tactile_calib = bool(args.tactile_calib)

    if args.async_policy and args.trex_refine_mode == "same-plan":
        mode = "same-plan"
    elif args.async_policy:
        mode = "async"
    else:
        mode = "sync"
    if args.trex_refine_mode == "same-plan" and not args.async_policy:
        warnings.append("--trex-refine-mode same-plan requires --async-policy; ignoring (running plain full-chunk)")
    trex_action_execution = bool(args.trex_action_execution and mode == "async")
    if args.trex_action_execution and mode == "sync":
        warnings.append("--trex-action-execution has no effect without --async-policy; ignoring")

    if mode == "same-plan":
        if info is None:
            warnings.append(
                "same-plan requested but server_info is unavailable: cannot verify the server runs --cascade; "
                "proceeding UNVERIFIED"
            )
        else:
            cascade = info.get("cascade")
            enabled = cascade.get("enabled") if isinstance(cascade, dict) else cascade
            if not enabled:
                raise ValueError("--trex-refine-mode same-plan requires the server to run with --cascade")
            force_delay_max = _section(info, "checkpoint").get("force_delay_max")
            if force_delay_max is not None and args.trex_max_refine_offset > int(force_delay_max):
                warnings.append(
                    f"--trex-max-refine-offset {args.trex_max_refine_offset} exceeds the checkpoint's "
                    f"force_delay_max {force_delay_max}: refines beyond it fall outside the delay-aug coverage"
                )

    tactile_info = _section(info, "tactile")
    if tactile_info.get("source") == "live" and not args.tactile:
        warnings.append(
            "the server re-reads its own live tactile stream per step but falls back to the request's maps when "
            "that stream is incomplete; with --no-tactile those would be zeros"
        )
    elif tactile_info.get("source") == "none" and spec.uses_tactile and not args.tactile:
        warnings.append(
            "the server reads no tactile stream of its own (--tactile-source none), so with --no-tactile the "
            "policy sees all-zero tactile inputs"
        )
    if info is not None and tactile_info.get("enabled") is False:
        logger.info(
            "server runs with --disable-tactile: the tactile keys are still sent (required_observation_keys says so)"
        )

    smoothing = SmoothingConfig(
        smooth_alpha=float(args.smooth_alpha),
        boundary_smooth_alpha=float(args.boundary_smooth_alpha),
        boundary_smooth_steps=max(0, int(args.boundary_smooth_steps)),
        boundary_blend_steps=max(0, int(args.boundary_blend_steps)),
        boundary_blend_alpha=float(args.boundary_blend_alpha),
    )
    return RunSettings(
        spec=spec,
        layout=layout,
        smoothing=smoothing,
        control_hz=control_hz,
        dt=1.0 / control_hz,
        execution_steps=execution_steps,
        steps=int(args.steps),
        tactile_live=bool(args.tactile),
        tactile_calib=tactile_calib,
        mode=mode,
        trex_action_execution=trex_action_execution,
        warnings=warnings,
    )


# ── init posture ─────────────────────────────────────────────────────────────


def load_init_state_yaml(path: str) -> dict:
    """The ``states`` dict of a ``compute_average_initial_state.py`` YAML."""
    with open(path, encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, dict) or "states" not in payload:
        raise ValueError(
            f"{path} has no top-level 'states' dict. Was it generated by compute_average_initial_state.py?"
        )
    logger.info(
        "Loaded init state from %s (first_n_frames=%s, num_episodes=%s).",
        path,
        payload.get("first_n_frames"),
        payload.get("num_episodes"),
    )
    return payload["states"]


def resolve_right_arm_hold(qpos: Sequence[float] | None, init_state: dict | None) -> np.ndarray | None:
    """The fixed right-arm pose of a single-arm deployment: the flag, else the YAML's rest pose, else None."""
    if qpos is not None:
        q = np.asarray(qpos, dtype=np.float64)
    elif init_state and init_state.get("right_arm", {}).get("qpos"):
        q = np.asarray(init_state["right_arm"]["qpos"], dtype=np.float64)
    else:
        return None
    if q.size != 7:
        raise ValueError(f"right-arm hold qpos must be 7 values, got {q.size}")
    return q


# ── tactile calibration ──────────────────────────────────────────────────────


def calibrate_tactile(args: argparse.Namespace, run: Callable = subprocess.run, sleep: Callable = time.sleep) -> bool:
    """Tare both hands' tactile through the robot-side calibration script, retrying.

    Runs ``docker exec -w /workspace <container> python3 calibrate_sharpa_tactile.py``
    over SSH; that script exits 0 only when both hands report a successful tare.
    Hands must be stationary with free fingertips. Key-based SSH, or
    ``THOR_SSH_PASS`` with ``sshpass`` installed.
    """
    target = args.tactile_calib_ssh or f"unitree@{args.tactile_host}"
    command = [
        "ssh",
        "-o",
        "BatchMode=yes",
        "-o",
        "ConnectTimeout=10",
        target,
        "docker",
        "exec",
        "-w",
        "/workspace",
        args.tactile_calib_container,
        "python3",
        "calibrate_sharpa_tactile.py",
    ]
    password = os.environ.get("THOR_SSH_PASS")
    if password:
        command = ["sshpass", "-p", password] + command[:1] + command[3:]  # sshpass needs password auth
    attempts = int(args.tactile_calib_attempts)
    for attempt in range(1, attempts + 1):
        logger.info("[tactile-calib] attempt %d/%d (keep hands stationary, fingertips free)...", attempt, attempts)
        try:
            # stdin=DEVNULL: otherwise ssh inherits and consumes the operator's pending keystrokes.
            result = run(command, capture_output=True, text=True, timeout=120, stdin=subprocess.DEVNULL)
        except (subprocess.TimeoutExpired, FileNotFoundError) as exc:
            logger.warning("[tactile-calib] ssh execution failed: %s", exc)
            sleep(1.0)
            continue
        for line in (result.stdout or "").splitlines():
            if line.strip().startswith("{"):
                logger.info("[tactile-calib] %s", line.strip())
        if result.returncode == 0:
            sleep(float(args.tactile_calib_settle_s))
            logger.info("[tactile-calib] TACTILE CALIBRATION COMPLETE (both hands)")
            return True
        tail = (result.stderr or "").strip().splitlines()[-1:] or ["(no stderr)"]
        logger.warning("[tactile-calib] failed (rc=%s): %s", result.returncode, tail[0])
        sleep(1.0)
    return False


def ensure_tactile_calibrated(args: argparse.Namespace, prompt: Callable[[str], str] = input) -> None:
    """Block until the tare succeeds, or the operator skips it or quits."""
    if not getattr(args, "tactile_calib", False):
        return
    while not calibrate_tactile(args):
        answer = (
            prompt(
                "[tactile-calib] calibration FAILED after retries. Check the robot-side producer / hand contact, "
                "then press [Enter] to retry, type 'skip' to proceed UNCALIBRATED, or 'q' to quit: "
            )
            .strip()
            .lower()
        )
        if answer == "skip":
            logger.warning("[tactile-calib] proceeding WITHOUT calibration (operator override)")
            return
        if answer == "q":
            raise SystemExit("aborted: tactile calibration failed")


# ── episode scoring ──────────────────────────────────────────────────────────


def prompt_episode_label(prompt: Callable[[str], str] = input) -> str:
    """One of ``success`` / ``fail`` / ``abort`` / ``retry`` from the operator; a closed stdin means abort."""
    while True:
        try:
            raw = prompt("\n[ablation] Episode outcome? [s]uccess / [f]ail / [a]bort / [r]etry: ").strip().lower()
        except EOFError:
            return "abort"
        if raw in LABEL_SHORTCUTS.values():
            return raw
        if raw in LABEL_SHORTCUTS:
            return LABEL_SHORTCUTS[raw]
        print(f"  unrecognised input '{raw}'. Type s, f, a, or r.")


def log_episode_result(path: str, *, episode_idx: int, tactile: bool, task: str, steps: int, label: str) -> None:
    """Append one JSON line; runs into the same file accumulate."""
    record = {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "episode_idx": episode_idx,
        "tactile": bool(tactile),
        "task": task,
        "steps": steps,
        "label": label,
    }
    with open(path, "a", encoding="utf-8") as handle:
        handle.write(json.dumps(record) + "\n")


# ── wiring ───────────────────────────────────────────────────────────────────


def make_robot(args: argparse.Namespace) -> RobotIO:
    """The robot the flags describe: ``FakeRobot`` with ``--fake-robot``, else a connected ``H2Robot``."""
    hands = HANDS if args.hands else ()
    if args.fake_robot:
        logger.info("Fake robot: synthetic cameras and joints, commands recorded.")
        return FakeRobot(arm=args.arm, hands=hands, binocular=(args.head_mode == "binocular"))
    options = RobotOptions(
        arm=args.arm,
        hands=hands,
        cameras=True,
        dds=DdsOptions(
            domain=args.dds_domain,
            network_interface=args.network_interface,
            network_address=args.network_address,
            network_peer=args.network_peer,
            body_peers=tuple(args.body_dds_peer or ()),
            discovery_ssh=args.body_dds_discovery_ssh,
            discovery_timeout_s=args.body_dds_discovery_timeout,
            discover_body_peers=bool(args.arm),
        ),
        arm_options=ArmOptions(
            motion=args.motion,
            head_pitch_home=args.head_pitch_home,
            velocity_limit=args.arm_velocity_limit,
            kp_low=args.kp_low,
            kp_wrist=args.kp_wrist,
            kd_low=args.kd_low,
            kd_wrist=args.kd_wrist,
            kp_head=args.kp_head,
            kd_head=args.kd_head,
        ),
        camera_options=CameraOptions(
            head_mode=args.head_mode,
            warmup_s=args.cam_warmup_s,
            network_address=args.camera_network_address,
            network_peer=args.camera_network_peer,
        ),
    )
    return H2Robot(options).connect()


def make_tactile_source(args: argparse.Namespace, spec: ObservationSpec) -> Any | None:
    """A live SharpaWave source for exactly the checkpoint's fingertips, or None.

    The stream is awaited for two seconds; a silent fingertip is a warning, not
    an error, because the old client never blocked on the stream either.
    """
    if not (args.tactile and spec.uses_tactile):
        return None
    from i4h_tasks.gr00t_n17_tactile.tactile.sharpa import SharpaLiveSource

    logger.info(
        "Connecting to the SharpaWave tactile stream at tcp://%s:%s (%d fingertips)...",
        args.tactile_host,
        args.tactile_port,
        len(spec.fingertips),
    )
    source = SharpaLiveSource(
        args.tactile_host,
        args.tactile_port,
        force_window=spec.force_window or DEFAULT_FORCE_WINDOW,
        fingertips=spec.fingertips,
    )
    try:
        source.wait_ready(2.0)
    except RuntimeError as exc:
        logger.warning("%s", exc)
    return source


def make_obs_fn(
    robot: RobotIO, spec: ObservationSpec, layout: CameraLayout, task: str, tactile_source: Any | None
) -> Callable[[], dict]:
    """The observation closure both the sync loop and the workers call.

    Reads the robot, the tactile source and builds the nested GR00T observation.
    Missing frames or fingertips are reported at most every few seconds; the
    observation still goes out (the policy's strict input check will say what
    it misses).
    """
    last_warning = {"video": 0.0, "tactile": 0.0}

    def warn(kind: str, message: str) -> None:
        now = time.monotonic()
        if now - last_warning[kind] >= WARN_EVERY_S:
            last_warning[kind] = now
            logger.warning("%s", message)

    def obs_fn() -> dict:
        snapshot = robot.snapshot(spec.hands)
        tactile = read_tactile(tactile_source, spec) if spec.uses_tactile else None
        if tactile is not None and tactile_source is not None and not tactile.complete:
            missing = getattr(tactile_source, "missing", lambda: [])()
            warn("tactile", f"tactile frame incomplete; sending zeros for {missing or 'every fingertip'}")
        obs = build_observation(snapshot, spec, layout, task, tactile)
        absent = [key for key in spec.video_keys if key not in obs.get("video", {})]
        if absent:
            warn("video", f"no camera frame for {absent}; the observation goes out without them")
        return obs

    return obs_fn


# ── check mode ───────────────────────────────────────────────────────────────


def _save_image(path: Path, image: np.ndarray, cv2: Any | None) -> Path:
    if cv2 is not None:
        cv2.imwrite(str(path), image)
        return path
    path = path.with_suffix(".npy")
    np.save(path, image)
    return path


def check_observations(args: argparse.Namespace) -> int:
    """Build one observation the way the loop would, print every leaf and save the frames."""
    settings = resolve_run_settings(args, None)
    for warning in settings.warnings:
        logger.warning("%s", warning)
    robot = make_robot(args)
    source = None
    try:
        source = make_tactile_source(args, settings.spec)
        obs = make_obs_fn(robot, settings.spec, settings.layout, args.task, source)()

        logger.info("=== Observation fields (after batch/time dims added) ===")
        for line in describe_observation(obs):
            logger.info("%s", line)
        if not obs.get("video"):
            logger.error("video dict is EMPTY -- no camera frames received.")

        try:
            import cv2
        except ImportError:
            cv2 = None
            logger.warning("cv2 is not installed; writing .npy arrays instead of PNGs")
        check_dir = Path(args.check_dir)
        check_dir.mkdir(parents=True, exist_ok=True)

        logger.info("=== Camera snapshots ===")
        for key, array in obs.get("video", {}).items():
            rgb = np.asarray(array)[0, 0]  # (1, 1, H, W, 3) -> (H, W, 3) RGB
            saved = _save_image(check_dir / f"check_{key}.png", np.ascontiguousarray(rgb[:, :, ::-1]), cv2)
            logger.info("  [%s] saved -> %s", key, saved)
        if obs.get("tactile"):
            logger.info("=== Tactile snapshots (%d fingertips) ===", len(obs["tactile"]))
            for key, array in obs["tactile"].items():
                frame = np.asarray(array)[0, 0]  # (1, 1, 240, 240) -> (240, 240)
                saved = _save_image(check_dir / f"check_tactile_{key}.png", frame, cv2)
                logger.info(
                    "  [%-25s] saved -> %s  nonzero_px=%6d  max=%3d",
                    key,
                    saved,
                    int((frame > 0).sum()),
                    int(frame.max()),
                )
        logger.info("Check complete.")
    finally:
        _quietly(robot.close, "robot close")
        if source is not None:
            _quietly(source.close, "tactile source close")
    return 0


# ── the control loop ─────────────────────────────────────────────────────────


def run_episode(
    robot: RobotIO,
    client: Any,
    worker: Any | None,
    settings: RunSettings,
    executor: ChunkExecutor,
    obs_fn: Callable[[], dict],
    tactile_source: Any | None,
    args: argparse.Namespace,
) -> int:
    """Execute ``settings.steps`` control steps from successive chunks; returns the steps executed.

    Per chunk: acquire it (per mode), map its capture time to a start step,
    hand it to the executor, walk it for ``horizon`` ticks sending one arm and
    one target per hand each tick, and schedule the T-Rex refines / tail
    refreshes between ticks.
    """
    spec, dt, mode = settings.spec, settings.dt, settings.mode
    steps, execution_steps = settings.steps, settings.execution_steps
    steps_executed = 0
    last_loaded_version = -1
    pending = None  # same-plan: the next plan's slow pass in flight
    pending_t_submit = 0.0
    refine_slot = None  # same-plan: the fast refine of the current plan in flight

    while steps_executed < steps:
        if mode == "same-plan":
            if pending is None:
                pending_t_submit = time.monotonic()
                pending = worker.submit("slow", obs_fn)
            if not pending.event.wait(CHUNK_WAIT_S):
                raise RuntimeError("T-Rex same-plan slow produced no chunk in 30s.")
            chunk = pending.chunk
            # Age off the worker's actual capture time; the FIFO can delay the
            # dequeue well past submit, which would overshoot start_step.
            t_obs = pending.t_obs or pending_t_submit
            pending = None
            refine_slot = None
            if chunk is None:
                continue
            chunk = copy_chunk(chunk)
            chunk_len = chunk_length(chunk, execution_steps)
            start_step = start_step_for(time.monotonic() - t_obs, dt, chunk_len)
        elif mode == "async":
            wait_started = time.monotonic()
            while True:
                latest = worker.get_latest()
                if latest is not None and latest.version > last_loaded_version:
                    break
                if worker.failure is not None:
                    raise RuntimeError(f"async inference stopped: {worker.failure}")
                if time.monotonic() - wait_started > CHUNK_WAIT_S:
                    raise RuntimeError("Async worker produced no chunk in 30s.")
                time.sleep(args.async_cache_poll_s)
            last_loaded_version = latest.version
            # The tail refresh writes in place, so it needs a private copy.
            chunk = copy_chunk(latest.chunk) if settings.trex_action_execution else latest.chunk
            chunk_len = chunk_length(chunk, execution_steps)
            start_step = start_step_for(time.monotonic() - latest.t_obs, dt, chunk_len)
        else:
            chunk, _ = client.get_action(obs_fn())
            chunk_len = chunk_length(chunk, execution_steps)
            start_step = 0

        horizon = plan_horizon(execution_steps, chunk_len, start_step, steps - steps_executed)
        if horizon <= 0:
            logger.warning("chunk skipped: start_step=%d chunk_len=%d", start_step, chunk_len)
            continue

        blend_window = executor.begin_chunk(chunk, start_step, horizon)
        if blend_window > 0:
            alpha = max(1e-6, settings.smoothing.boundary_blend_alpha)
            logger.info(
                "boundary blend: %d steps (old @ %s.. x new @ %d.., alpha=%.2f, w_old: 1.000->%.3f)",
                blend_window,
                getattr(executor, "prev_chunk_next_t", "?"),
                start_step,
                alpha,
                math.exp(-alpha * (blend_window - 1) / blend_window),
            )

        for offset in range(horizon):
            t = start_step + offset
            t0 = time.perf_counter()
            targets = executor.step(offset)
            if robot.has_arm and targets.left_arm is not None and targets.right_arm is not None:
                robot.send_arm(np.concatenate([targets.left_arm, targets.right_arm]))
            for side in robot.hands:
                q = targets.left_hand if side == "left" else targets.right_hand
                if q is not None:
                    robot.send_hand(side, q, blocking=False)
            pace(t0, dt)
            steps_executed += 1

            if mode == "same-plan":
                # Apply a finished refine of the current plan index-for-index
                # from t+1 on, so the step just sent stands.
                if refine_slot is not None and refine_slot.event.is_set():
                    fresh = refine_slot.chunk
                    refine_slot = None
                    if fresh is not None:
                        blend = args.trex_refine_blend_alpha if args.trex_refine_blend else None
                        n = apply_same_plan_refine(executor.chunk, fresh, t, blend)
                        if n:
                            how = (
                                f"blend a={args.trex_refine_blend_alpha:.2f}" if args.trex_refine_blend else "overwrite"
                            )
                            logger.info("trex same-plan refine applied @t=%d: %d steps (%s)", t, n, how)
                if refine_due(
                    offset, t, horizon, args.trex_refine_every, args.trex_max_refine_offset, refine_slot is None
                ):
                    if args.trex_contact_gate:
                        contact = read_contact_metric(tactile_source, spec.fingertips, "left")
                        if contact < args.trex_contact_threshold:
                            logger.info(
                                "trex refine gated OFF @t=%d: contact %.2f < %s",
                                t,
                                contact,
                                args.trex_contact_threshold,
                            )
                        else:
                            refine_slot = worker.submit("refine", obs_fn)
                    else:
                        refine_slot = worker.submit("refine", obs_fn)
                # Predict the next plan a fixed lead before the boundary; FIFO
                # runs the refines queued above before this slow pass.
                if slow_due(offset, horizon, args.trex_slow_lead, pending is not None):
                    pending_t_submit = time.monotonic()
                    pending = worker.submit("slow", obs_fn)
            elif (
                settings.trex_action_execution
                and worker is not None
                and refresh_due(offset, horizon, args.trex_refine_every)
            ):
                peek = worker.get_latest()
                if peek is not None and peek.version > last_loaded_version:
                    fresh_now = int(max(0.0, time.monotonic() - peek.t_obs) / dt)
                    n = apply_new_plan_refresh(executor.chunk, peek.chunk, t + 1, fresh_now + 1)
                    last_loaded_version = peek.version
                    if n:
                        logger.info("trex tail refresh @offset %d: %d steps (v%d)", offset, n, peek.version)

        if mode == "same-plan" and pending is None and steps_executed < steps:
            pending_t_submit = time.monotonic()
            pending = worker.submit("slow", obs_fn)

        logger.info("chunk done (%d steps from start_step=%d), total=%d/%d", horizon, start_step, steps_executed, steps)
        executor.end_chunk()

    return steps_executed


def _quietly(action: Callable[[], Any], what: str) -> None:
    """Run one teardown step so a failure (or a second Ctrl+C) never skips the next."""
    try:
        action()
    except KeyboardInterrupt:
        logger.warning("%s interrupted", what)
    except Exception as exc:
        logger.warning("%s failed: %s", what, exc)


def run(args: argparse.Namespace) -> int:
    """Connect to the robot and the server, then run episodes until told to stop."""
    robot: RobotIO | None = None
    client: GR00TClient | None = None
    worker: Any | None = None
    tactile_source: Any | None = None
    started = False
    label_counts = {"success": 0, "fail": 0, "abort": 0}
    try:
        robot = make_robot(args)
        init_state = load_init_state_yaml(args.init_state_yaml) if args.init_state_yaml else None
        right_arm_hold = resolve_right_arm_hold(args.right_arm_hold_qpos, init_state)
        home_and_hold(robot, args.hold_s)
        init_and_hold(robot, init_state, right_arm_hold, args.init_state_hold_s)
        if right_arm_hold is None and robot.has_arm:
            # Per-tick hold target for chunks without right_arm (single-arm policy).
            right_arm_hold = np.asarray(robot.arm_q(), dtype=np.float64)[7:14]

        logger.info("Connecting to the policy server at %s:%s ...", args.policy_host, args.policy_port)
        client = GR00TClient(host=args.policy_host, port=args.policy_port, timeout_ms=args.timeout_ms)
        wait_for_server(client)
        info = client.server_info()
        if info is not None:
            logger.info("%s", describe_server_info(info))
        settings = resolve_run_settings(args, info)
        for warning in settings.warnings:
            logger.warning("%s", warning)
        for note in (info or {}).get("notes") or []:
            logger.info("server note: %s", note)
        args.tactile_calib = settings.tactile_calib

        tactile_source = make_tactile_source(args, settings.spec)
        # Tare at the init pose before the policy ever engages; repeated per episode below.
        ensure_tactile_calibrated(args)

        answer = input("\nPress [Enter] to start inference (or 'q' to quit): ")
        if answer.strip().lower() == "q":
            return 0
        started = True
        logger.info(
            "Inference config: %.1f Hz, execution_steps=%d, steps_per_episode=%d, mode=%s, task=%r",
            settings.control_hz,
            settings.execution_steps,
            settings.steps,
            settings.mode,
            args.task,
        )
        logger.info("Running: home(%ds) -> inference -> home(%ds) -> ...  Ctrl+C to stop.", args.hold_s, args.hold_s)

        obs_fn = make_obs_fn(robot, settings.spec, settings.layout, args.task, tactile_source)
        executor = ChunkExecutor(settings.smoothing, right_arm_hold)
        if settings.mode == "same-plan":
            worker = TRexPlanWorker(client)
            worker.start()
            logger.info("T-Rex same-plan worker started (client-driven slow/fast against the server cascade).")
        elif settings.mode == "async":
            worker = AsyncPolicyWorker(client, obs_fn, log_every=30)
            worker.start()
            logger.info("Async policy worker started (real-time chunking).")

        episode = 0
        while True:
            episode += 1
            if worker is not None:
                # The worker owns the socket; reset through it, then drop any
                # chunk produced before this episode's reset.
                worker.pause()
                worker.request_reset()
                worker.resume()
            else:
                client.reset()
            logger.info("=== Episode %d ===", episode)
            if episode > 1:
                ensure_tactile_calibrated(args)
            executor.reset()

            executed = run_episode(robot, client, worker, settings, executor, obs_fn, tactile_source, args)

            if worker is not None:
                worker.pause()
            logger.info("Episode %d complete (%d steps). Going home...", episode, executed)
            home_and_hold(robot, args.hold_s)
            init_and_hold(robot, init_state, right_arm_hold, args.init_state_hold_s)

            if args.log_results:
                label = prompt_episode_label()
                if label == "retry":
                    logger.info("  -> retry: re-running episode %d (not logged)", episode)
                    episode -= 1
                    continue
                label_counts[label] += 1
                log_episode_result(
                    args.log_results,
                    episode_idx=episode,
                    tactile=settings.tactile_live,
                    task=args.task,
                    steps=settings.steps,
                    label=label,
                )
                logger.info("  -> episode %d logged as '%s' to %s", episode, label, args.log_results)
                if label == "abort":
                    logger.info("  -> abort requested; ending run.")
                    break
            if args.num_episodes is not None and episode >= args.num_episodes:
                logger.info("=== Reached --num-episodes %d; ending run. ===", args.num_episodes)
                break
    except KeyboardInterrupt:
        logger.info("Stopped by user.")
    finally:
        # Stop the worker before touching hardware so no chunk arrives mid-teardown.
        if worker is not None:
            _quietly(worker.stop, "worker stop")
        if robot is not None:
            if started:
                _quietly(lambda: home_and_hold(robot, args.hold_s), "go home")
            _quietly(robot.close, "robot close")
        if tactile_source is not None:
            _quietly(tactile_source.close, "tactile source close")
        if client is not None:
            _quietly(client.close, "client close")
        if args.log_results:
            _log_summary(args, label_counts)
        logger.info("Done.")
    return 0


def _log_summary(args: argparse.Namespace, label_counts: dict[str, int]) -> None:
    total = sum(label_counts.values())
    if total == 0:
        logger.info("=== No episodes logged to %s. ===", args.log_results)
        return
    logger.info(
        "=== Ablation run summary ===\n  arm:           %s\n  log file:      %s\n  total:         %d\n"
        "  success:       %d\n  fail:          %d\n  abort:         %d\n  success rate:  %.1f%%",
        "tactile" if args.tactile else "no-tactile",
        args.log_results,
        total,
        label_counts["success"],
        label_counts["fail"],
        label_counts["abort"],
        100.0 * label_counts["success"] / total,
    )


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO, format="[gr00t-n17-tactile-deploy] %(message)s"
    )
    try:
        if args.check:
            return check_observations(args)
        return run(args)
    except (ValueError, FileNotFoundError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    sys.exit(main())

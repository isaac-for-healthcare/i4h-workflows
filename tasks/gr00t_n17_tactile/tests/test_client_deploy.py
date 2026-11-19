# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The deployment CLI: flags, server-driven settings, operator helpers and the control loop on fakes."""

from __future__ import annotations

import json
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from i4h_tasks.gr00t_n17_tactile.client import deploy, observation  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.client.execution import (  # noqa: E402
    AsyncPolicyWorker,
    ChunkExecutor,
    SmoothingConfig,
    TRexPlanWorker,
)
from i4h_tasks.gr00t_n17_tactile.client.robot import FakeRobot  # noqa: E402
from i4h_tasks.gr00t_n17_tactile.tactile.interface import FINGERTIPS  # noqa: E402

# ── fixtures ─────────────────────────────────────────────────────────────────

#: Shaped like ``server.py:served_info()`` for the bimanual force checkpoint, with
#: non-default rate and execution length so the tests can tell server values apart.
SERVED_INFO = {
    "task": "gr00t_n17_tactile/h2_sharpa_tactile",
    "embodiment": "new_embodiment",
    "video_modality_keys": ["high", "left_wrist_view", "right_wrist_view"],
    "state_groups": {"left_arm": [0, 7], "right_arm": [7, 14], "left_hand": [14, 36], "right_hand": [36, 58]},
    "action_horizon": 32,
    "execution_steps": 12,
    "control_hz": 25.0,
    "tactile": {
        "enabled": True,
        "source": "live",
        "fingertips": list(FINGERTIPS),
        "num_tokens": 10,
        "feature_dim": 512,
        "force_window": 16,
        "delay_aug": True,
        "refresh": "per-step",
    },
    "cascade": {"enabled": False, "fast_loops": 2},
    "checkpoint": {
        "class": "Gr00tN1d7Tactile",
        "use_tactile": True,
        "use_tactile_force": True,
        "delay_aug": True,
        "tau_split": 0.5,
        "force_delay_max": 16,
        "num_tactile_tokens": 10,
    },
    "required_observation_keys": {
        "video": ["high", "left_wrist_view", "right_wrist_view"],
        "state": ["left_arm", "right_arm", "left_hand", "right_hand"],
        "language": ["annotation.human.task_description"],
        "tactile": [f"{name}_deform" for name in FINGERTIPS],
        "tactile_force": ["force"],
    },
}


def _info(**changes):
    info = json.loads(json.dumps(SERVED_INFO))
    for path, value in changes.items():
        node = info
        *parents, leaf = path.split(".")
        for key in parents:
            node = node[key]
        node[leaf] = value
    return info


def _args(*argv: str):
    return deploy.build_parser().parse_args(list(argv))


def _settings(mode: str, *, steps: int, execution_steps: int, trex_action_execution: bool = False):
    spec = observation.default_spec(tactile=False, force_window=0)
    layout = observation.resolve_layout(spec, "binocular", None, True)
    return deploy.RunSettings(
        spec=spec,
        layout=layout,
        smoothing=SmoothingConfig(),
        # Pacing is monkeypatched out in the loop tests; a long tick keeps start_step at 0 whatever the host does.
        control_hz=0.1,
        dt=10.0,
        execution_steps=execution_steps,
        steps=steps,
        tactile_live=False,
        tactile_calib=False,
        mode=mode,
        trex_action_execution=trex_action_execution,
        warnings=[],
    )


def _ramp_chunk(k: int, horizon: int) -> dict[str, np.ndarray]:
    """Row ``t`` of chunk ``k`` is ``100 k + t`` on every arm joint (+0.5 on the right arm) and hand joint."""
    t = np.arange(horizon, dtype=np.float32)[None, :, None]
    return {
        "left_arm": np.repeat(100.0 * k + t, 7, axis=2),
        "right_arm": np.repeat(100.0 * k + t + 0.5, 7, axis=2),
        "left_hand": np.repeat(100.0 * k + t, 22, axis=2),
        "right_hand": np.repeat(100.0 * k + t, 22, axis=2),
    }


class FakePolicy:
    """Returns ramp chunks numbered by call and records every ``options`` it was given."""

    def __init__(self, horizon: int = 8, latency: float = 0.0) -> None:
        self.horizon = horizon
        #: Seconds each get_action blocks, as inference does; keeps a free-running worker from spinning.
        self.latency = latency
        self.calls = 0
        self.resets = 0
        self.options: list = []
        self._lock = threading.Lock()

    def get_action(self, observation, options=None):
        with self._lock:
            self.calls += 1
            k = self.calls
            self.options.append(options)
        assert set(observation) >= {"video", "state", "language"}
        if self.latency:
            time.sleep(self.latency)
        return _ramp_chunk(k, self.horizon), {}

    def reset(self, options=None):
        self.resets += 1


# ── the parser ───────────────────────────────────────────────────────────────

PARSER_DEFAULTS = {
    "policy_host": "localhost",
    "policy_port": 5555,
    "timeout_ms": 15000,
    "task": "",
    "check": False,
    "check_dir": "/tmp",
    "fake_robot": False,
    "frequency": None,
    "action_horizon": None,
    "steps": 30,
    "num_episodes": None,
    "log_results": None,
    "hold_s": 20,
    "init_state_yaml": None,
    "init_state_hold_s": 3,
    "right_arm_hold_qpos": None,
    "async_policy": False,
    "async_cache_poll_s": 0.005,
    "smooth_alpha": 1.0,
    "boundary_smooth_alpha": 1.0,
    "boundary_smooth_steps": 0,
    "boundary_blend_steps": 0,
    "boundary_blend_alpha": 2.0,
    "trex_action_execution": False,
    "trex_refine_every": 4,
    "trex_refine_mode": "new-plan",
    "trex_max_refine_offset": 4,
    "trex_slow_lead": 3,
    "trex_contact_gate": False,
    "trex_contact_threshold": 4.0,
    "trex_refine_blend": False,
    "trex_refine_blend_alpha": 0.5,
    "arm": True,
    "hands": True,
    "motion": False,
    "head_pitch_home": 0.6,
    "arm_velocity_limit": 20.0,
    "kp_low": None,
    "kp_wrist": None,
    "kd_low": None,
    "kd_wrist": None,
    "kp_head": None,
    "kd_head": None,
    "dds_domain": 0,
    "network_interface": None,
    "network_address": None,
    "network_peer": None,
    "body_dds_peer": None,
    "body_dds_discovery_ssh": None,
    "body_dds_discovery_timeout": 35.0,
    "head_mode": "mono",
    "head_eye": None,
    "cam_warmup_s": 5.0,
    "swap_wrist_cams": False,
    "camera_network_address": None,
    "camera_network_peer": None,
    "wrist_cams": "auto",
    "state_hands": "auto",
    "checkpoint_tactile": None,
    "checkpoint_tactile_force": None,
    "tactile": True,
    "tactile_host": "127.0.0.1",
    "tactile_port": 7779,
    "tactile_calib": None,
    "tactile_calib_ssh": None,
    "tactile_calib_container": "sharpa180",
    "tactile_calib_attempts": 3,
    "tactile_calib_settle_s": 1.0,
    "verbose": False,
}


def test_parser_defaults_are_exactly_the_kept_flags() -> None:
    assert vars(_args()) == PARSER_DEFAULTS
    assert deploy.build_parser().prog == "i4h-tasks-gr00t-n17-tactile-deploy"


@pytest.mark.parametrize(
    "flag,dest",
    [
        ("--no-arm", "arm"),
        ("--no-hands", "hands"),
        ("--no-tactile", "tactile"),
        ("--no-checkpoint-tactile", "checkpoint_tactile"),
        ("--no-checkpoint-tactile-force", "checkpoint_tactile_force"),
        ("--no-tactile-calib", "tactile_calib"),
    ],
)
def test_negative_bool_spellings_carry_over(flag: str, dest: str) -> None:
    assert getattr(_args(flag), dest) is False
    assert getattr(_args(flag.replace("--no-", "--")), dest) is True


def test_repeatable_and_multi_value_flags() -> None:
    args = _args(
        "--body-dds-peer", "10.0.0.1:7400", "--body-dds-peer", "10.0.0.1:7401", "--right-arm-hold-qpos", *["0.1"] * 7
    )
    assert args.body_dds_peer == ["10.0.0.1:7400", "10.0.0.1:7401"]
    assert args.right_arm_hold_qpos == [0.1] * 7
    with pytest.raises(SystemExit):
        _args("--right-arm-hold-qpos", "0.1", "0.2")


@pytest.mark.parametrize(
    "argv",
    [
        ["--image-source", "ros"],
        ["--img-server-ip", "192.0.2.1"],
        ["--hands-source", "dds"],
        ["--hand-speed-coeff", "0.5"],
        ["--hand-current-coeff", "0.6"],
        ["--hand-smooth-alpha", "0.5"],
        ["--hand-reanchor"],
        ["--record-run", "/tmp/x"],
        ["--async-boundary-lead", "2"],
        ["--video-history-hz", "10"],
        ["--replay-dataset", "/tmp/x"],
        ["--allow-legacy-server"],
    ],
)
def test_dropped_flags_are_rejected(argv: list[str]) -> None:
    with pytest.raises(SystemExit) as exc:
        _args(*argv)
    assert exc.value.code == 2


# ── settings from server_info ────────────────────────────────────────────────


def test_settings_follow_the_served_checkpoint() -> None:
    settings = deploy.resolve_run_settings(_args(), _info())
    assert settings.control_hz == 25.0 and settings.dt == pytest.approx(1 / 25.0)
    assert settings.execution_steps == 12
    assert settings.steps == 30
    assert settings.mode == "sync" and settings.trex_action_execution is False
    assert settings.tactile_live is True and settings.tactile_calib is True
    spec = settings.spec
    assert spec.video_keys == ("high", "left_wrist_view", "right_wrist_view")
    assert spec.state_keys == ("left_arm", "right_arm", "left_hand", "right_hand")
    assert spec.fingertips == FINGERTIPS and spec.force_window == 16
    assert settings.layout == observation.CameraLayout("mono", "left", False)
    assert settings.warnings == []


def test_explicit_rate_and_horizon_win_with_a_warning_past_the_chunk() -> None:
    settings = deploy.resolve_run_settings(_args("--frequency", "10", "--action-horizon", "40"), _info())
    assert settings.control_hz == 10.0 and settings.execution_steps == 40
    assert any("action_horizon" in w for w in settings.warnings)


def test_same_plan_needs_the_server_cascade() -> None:
    argv = ("--async-policy", "--trex-refine-mode", "same-plan")
    with pytest.raises(ValueError, match="cascade"):
        deploy.resolve_run_settings(_args(*argv), _info())
    settings = deploy.resolve_run_settings(_args(*argv), _info(**{"cascade.enabled": True}))
    assert settings.mode == "same-plan" and settings.warnings == []
    wide = deploy.resolve_run_settings(
        _args(*argv, "--trex-max-refine-offset", "20"), _info(**{"cascade.enabled": True})
    )
    assert any("force_delay_max" in w for w in wide.warnings)
    unverified = deploy.resolve_run_settings(_args(*argv), None)
    assert unverified.mode == "same-plan" and any("UNVERIFIED" in w for w in unverified.warnings)


def test_same_plan_without_async_falls_back_with_a_warning() -> None:
    settings = deploy.resolve_run_settings(_args("--trex-refine-mode", "same-plan"), _info())
    assert settings.mode == "sync"
    assert any("requires --async-policy" in w for w in settings.warnings)


def test_trex_action_execution_only_counts_in_async_mode() -> None:
    plain = deploy.resolve_run_settings(_args("--trex-action-execution"), _info())
    assert plain.trex_action_execution is False and any("--trex-action-execution" in w for w in plain.warnings)
    asynchronous = deploy.resolve_run_settings(_args("--trex-action-execution", "--async-policy"), _info())
    assert asynchronous.mode == "async" and asynchronous.trex_action_execution is True


def test_tactile_calib_auto_and_live_source_warning() -> None:
    off = deploy.resolve_run_settings(_args("--no-tactile"), _info())
    assert off.tactile_live is False and off.tactile_calib is False
    assert any("live" in w for w in off.warnings)
    forced = deploy.resolve_run_settings(_args("--no-tactile", "--tactile-calib"), _info())
    assert forced.tactile_calib is True
    # Against a server without its own stream (the --cascade case) the client is the only tactile input.
    alone = deploy.resolve_run_settings(_args("--no-tactile"), _info(**{"tactile.source": "none"}))
    assert any("all-zero tactile" in w for w in alone.warnings)
    assert not any("all-zero tactile" in w for w in deploy.resolve_run_settings(_args(), _info()).warnings)
    vision_only = deploy.resolve_run_settings(
        _args(),
        _info(
            **{"required_observation_keys": {}, "checkpoint.use_tactile": False, "checkpoint.use_tactile_force": False}
        ),
    )
    assert vision_only.spec.uses_tactile is False and vision_only.tactile_calib is False


def test_legacy_defaults_without_server_info() -> None:
    settings = deploy.resolve_run_settings(_args(), None)
    assert settings.control_hz == 30.0 and settings.execution_steps == 16
    spec = settings.spec
    assert spec.uses_tactile and not spec.uses_force
    assert "right_wrist_view" in spec.video_keys and "left_wrist_view" not in spec.video_keys
    assert spec.state_keys == ("left_arm", "right_arm", "left_hand", "right_hand")
    assert any("server_info unavailable" in w for w in settings.warnings)


def test_force_without_tactile_is_a_config_error() -> None:
    with pytest.raises(ValueError):
        deploy.resolve_run_settings(_args("--no-checkpoint-tactile", "--checkpoint-tactile-force"), _info())


# ── init posture ─────────────────────────────────────────────────────────────


def test_init_state_yaml_needs_states(tmp_path: Path) -> None:
    good = tmp_path / "init.yaml"
    good.write_text("first_n_frames: 10\nnum_episodes: 3\nstates:\n  left_arm:\n    qpos: [1, 2, 3, 4, 5, 6, 7]\n")
    assert deploy.load_init_state_yaml(str(good)) == {"left_arm": {"qpos": [1, 2, 3, 4, 5, 6, 7]}}
    bad = tmp_path / "bad.yaml"
    bad.write_text("left_arm:\n  qpos: [1]\n")
    with pytest.raises(ValueError, match="states"):
        deploy.load_init_state_yaml(str(bad))


def test_right_arm_hold_precedence_and_size() -> None:
    init_state = {"right_arm": {"qpos": [0.2] * 7}}
    assert deploy.resolve_right_arm_hold([0.1] * 7, init_state).tolist() == [0.1] * 7
    assert deploy.resolve_right_arm_hold(None, init_state).tolist() == [0.2] * 7
    assert deploy.resolve_right_arm_hold(None, None) is None
    assert deploy.resolve_right_arm_hold(None, {"right_arm": {"qpos": []}}) is None
    with pytest.raises(ValueError, match="7 values"):
        deploy.resolve_right_arm_hold([0.1] * 6, None)


# ── tactile calibration ──────────────────────────────────────────────────────


class _Run:
    def __init__(self, codes: list[int]) -> None:
        self.codes = list(codes)
        self.commands: list[list[str]] = []
        self.kwargs: list[dict] = []

    def __call__(self, command, **kwargs):
        self.commands.append(list(command))
        self.kwargs.append(kwargs)
        return subprocess.CompletedProcess(command, self.codes.pop(0), stdout='{"ok": true}\n', stderr="boom\n")


def test_calibrate_tactile_success_settles(monkeypatch) -> None:
    monkeypatch.delenv("THOR_SSH_PASS", raising=False)
    run, sleeps = _Run([0]), []
    args = _args("--tactile-host", "bridge.example", "--tactile-calib-settle-s", "0.25")
    assert deploy.calibrate_tactile(args, run=run, sleep=sleeps.append) is True
    assert len(run.commands) == 1 and sleeps == [0.25]
    command = run.commands[0]
    assert command[:5] == ["ssh", "-o", "BatchMode=yes", "-o", "ConnectTimeout=10"]
    assert command[5] == "unitree@bridge.example"
    assert command[6:] == ["docker", "exec", "-w", "/workspace", "sharpa180", "python3", "calibrate_sharpa_tactile.py"]
    assert run.kwargs[0] == {"capture_output": True, "text": True, "timeout": 120, "stdin": subprocess.DEVNULL}


def test_calibrate_tactile_retries_then_gives_up(monkeypatch) -> None:
    monkeypatch.setenv("THOR_SSH_PASS", "secret")
    run, sleeps = _Run([1, 1, 1]), []
    args = _args("--tactile-calib-ssh", "op@host.example", "--tactile-calib-attempts", "3")
    assert deploy.calibrate_tactile(args, run=run, sleep=sleeps.append) is False
    assert len(run.commands) == 3 and sleeps == [1.0, 1.0, 1.0]
    command = run.commands[0]
    assert command[:3] == ["sshpass", "-p", "secret"] and command[3] == "ssh"
    assert "BatchMode=yes" not in command and "op@host.example" in command


def test_ensure_tactile_calibrated_operator_override(monkeypatch) -> None:
    calls = []
    monkeypatch.setattr(deploy, "calibrate_tactile", lambda args: calls.append(1) is None and False)
    args = _args()
    args.tactile_calib = False
    deploy.ensure_tactile_calibrated(args, prompt=lambda _: pytest.fail("no prompt when calibration is off"))
    assert calls == []
    args.tactile_calib = True
    answers = iter(["", "skip"])
    deploy.ensure_tactile_calibrated(args, prompt=lambda _: next(answers))
    assert len(calls) == 2
    with pytest.raises(SystemExit):
        deploy.ensure_tactile_calibrated(args, prompt=lambda _: "q")


# ── episode scoring ──────────────────────────────────────────────────────────


def test_prompt_episode_label_shortcuts_and_eof(capsys) -> None:
    answers = iter(["x", "S", "fail", "r"])
    assert deploy.prompt_episode_label(prompt=lambda _: next(answers)) == "success"
    assert "unrecognised" in capsys.readouterr().out
    assert deploy.prompt_episode_label(prompt=lambda _: next(answers)) == "fail"
    assert deploy.prompt_episode_label(prompt=lambda _: next(answers)) == "retry"

    def closed(_):
        raise EOFError

    assert deploy.prompt_episode_label(prompt=closed) == "abort"


def test_log_episode_result_appends_jsonl(tmp_path: Path) -> None:
    path = tmp_path / "results.jsonl"
    deploy.log_episode_result(str(path), episode_idx=1, tactile=True, task="pick", steps=30, label="success")
    deploy.log_episode_result(str(path), episode_idx=2, tactile=False, task="pick", steps=30, label="fail")
    records = [json.loads(line) for line in path.read_text().splitlines()]
    assert len(records) == 2
    assert set(records[0]) == {"timestamp", "episode_idx", "tactile", "task", "steps", "label"}
    assert records[1]["tactile"] is False and records[1]["label"] == "fail"


# ── wiring on fakes ──────────────────────────────────────────────────────────


def test_make_robot_builds_the_fake_from_flags() -> None:
    robot = deploy.make_robot(_args("--fake-robot", "--no-arm", "--no-hands", "--head-mode", "mono"))
    assert isinstance(robot, FakeRobot)
    assert robot.has_arm is False and robot.hands == ()
    assert robot.frames()["head"].shape == (480, 640, 3)
    both = deploy.make_robot(_args("--fake-robot"))
    assert both.has_arm is True and both.hands == ("left", "right")
    # The default head mode is mono: the three-camera launcher publishes /head/left only.
    assert both.frames()["head"].shape == (480, 640, 3)
    stereo = deploy.make_robot(_args("--fake-robot", "--head-mode", "binocular"))
    assert stereo.frames()["head"].shape == (480, 1280, 3)


def test_mono_head_mode_warns_when_the_checkpoint_wants_eye_views() -> None:
    info = {
        "required_observation_keys": {
            "video": ["left_eye_view", "right_eye_view"],
            "state": ["left_arm", "right_arm", "left_hand", "right_hand"],
            "language": ["annotation.human.task_description"],
        },
        "checkpoint": {"use_tactile": False, "use_tactile_force": False},
    }
    mono = deploy.resolve_run_settings(_args("--fake-robot"), info)
    assert mono.layout.head_mode == "mono"
    assert any("--head-mode binocular" in warning for warning in mono.warnings)
    stereo = deploy.resolve_run_settings(_args("--fake-robot", "--head-mode", "binocular"), info)
    assert stereo.layout.head_mode == "binocular"
    assert not any("--head-mode" in warning for warning in stereo.warnings)


def test_make_robot_maps_the_flags_onto_the_h2_options(monkeypatch) -> None:
    """``--no-arm`` turns body-peer discovery off; the gains and camera network flags reach the adapter."""
    built = []

    class _H2:
        def __init__(self, options) -> None:
            built.append(options)

        def connect(self):
            return self

    monkeypatch.setattr(deploy, "H2Robot", _H2)
    deploy.make_robot(_args("--no-arm", "--kp-low", "120", "--camera-network-address", "10.2.2.1"))
    options = built[-1]
    assert options.arm is False and options.dds.discover_body_peers is False and options.hands == ("left", "right")
    assert options.arm_options.kp_low == 120.0 and options.camera_options.network_address == "10.2.2.1"
    deploy.make_robot(_args("--no-hands", "--body-dds-peer", "10.1.1.4:7410"))
    options = built[-1]
    assert options.arm is True and options.dds.discover_body_peers is True and options.hands == ()
    assert options.dds.body_peers == ("10.1.1.4:7410",)


def test_obs_fn_routes_the_fake_streams_through_the_spec() -> None:
    robot = FakeRobot(frame_hw=(8, 8))
    spec = observation.default_spec(tactile=False)
    layout = observation.resolve_layout(spec, "binocular", None, True)
    obs = deploy.make_obs_fn(robot, spec, layout, "pick", None)()
    assert set(obs["video"]) == {"high", "left_wrist_view", "right_wrist_view"}
    assert obs["video"]["high"].shape == (1, 1, 8, 8, 3)
    # Swap on: right_wrist_view carries the left-wrist stream (fill 10), high the left eye (fill 30).
    assert obs["video"]["right_wrist_view"][0, 0, 1, 1].tolist() == [10, 10, 10]
    assert obs["video"]["left_wrist_view"][0, 0, 1, 1].tolist() == [20, 20, 20]
    assert obs["video"]["high"][0, 0, 1, 1].tolist() == [30, 30, 30]
    assert list(obs["state"]) == ["left_arm", "right_arm", "left_hand", "right_hand"]
    assert obs["state"]["left_hand"].shape == (1, 1, 22)
    assert obs["language"] == {"annotation.human.task_description": [["pick"]]}
    assert "tactile" not in obs and "tactile_force" not in obs


def test_obs_fn_sends_zero_tactile_without_a_source() -> None:
    robot = FakeRobot(frame_hw=(8, 8), hands=("left",))
    spec = observation.default_spec(tactile=True, force_window=16, hands=("left",), wrist_cams="right")
    layout = observation.resolve_layout(spec, "binocular", None, True)
    obs = deploy.make_obs_fn(robot, spec, layout, "pick", None)()
    assert list(obs["tactile"]) == [f"{name}_deform" for name in FINGERTIPS[:5]]
    assert obs["tactile"]["left_thumb_deform"].shape == (1, 1, 240, 240)
    assert obs["tactile_force"]["force"].shape == (1, 16, 30)
    assert list(obs["state"]) == ["left_arm", "right_arm", "left_hand"]


def test_check_mode_writes_one_file_per_frame(tmp_path: Path) -> None:
    # --check never reaches the server, so the legacy wrist default ("right") applies unless overridden.
    argv = ["--check", "--fake-robot", "--no-tactile", "--check-dir", str(tmp_path)]
    argv += ["--head-mode", "binocular", "--head-eye", "both"]
    assert deploy.main(argv + ["--wrist-cams", "both"]) == 0
    names = {path.name.rsplit(".", 1)[0] for path in tmp_path.iterdir()}
    assert names == {
        "check_left_eye_view",
        "check_right_eye_view",
        "check_left_wrist_view",
        "check_right_wrist_view",
        *(f"check_tactile_{name}_deform" for name in FINGERTIPS),
    }


def test_main_reports_config_errors_as_exit_2(capsys) -> None:
    code = deploy.main(["--check", "--fake-robot", "--no-checkpoint-tactile", "--checkpoint-tactile-force"])
    assert code == 2
    assert "checkpoint_tactile" in capsys.readouterr().err


# ── the control loop ─────────────────────────────────────────────────────────


@pytest.fixture
def no_pacing(monkeypatch):
    monkeypatch.setattr(deploy, "pace", lambda t0, dt: None)


def _run(robot, policy, worker, settings, args=None):
    args = args or _args()
    executor = ChunkExecutor(settings.smoothing, None)
    executor.reset()
    spec, layout = settings.spec, settings.layout
    obs_fn = deploy.make_obs_fn(robot, spec, layout, "pick", None)
    return deploy.run_episode(robot, policy, worker, settings, executor, obs_fn, None, args)


def test_sync_episode_walks_two_chunks_exactly(no_pacing) -> None:
    robot = FakeRobot(frame_hw=(8, 8), lag=1.0)
    policy = FakePolicy(horizon=8)
    executed = _run(robot, policy, None, _settings("sync", steps=8, execution_steps=4))
    assert executed == 8 and policy.calls == 2
    assert len(robot.sent_arm) == 8
    for i, row in enumerate(robot.sent_arm):
        k, t = 1 + i // 4, i % 4
        assert row[:7].tolist() == [100.0 * k + t] * 7
        assert row[7:].tolist() == [100.0 * k + t + 0.5] * 7
    assert len(robot.sent_hands["left"]) == 8 and len(robot.sent_hands["right"]) == 8
    assert robot.sent_hands["right"][-1].tolist() == [203.0] * 22
    # The fake tracks the last target completely with lag 1.0.
    assert robot.arm_q()[:7].tolist() == [203.0] * 7
    assert all(options is None for options in policy.options)


def test_sync_episode_holds_the_right_arm_for_a_single_arm_chunk(no_pacing) -> None:
    class LeftOnly(FakePolicy):
        def get_action(self, observation, options=None):
            chunk, info = super().get_action(observation, options)
            return {"left_arm": chunk["left_arm"], "left_hand": chunk["left_hand"]}, info

    robot = FakeRobot(frame_hw=(8, 8))
    settings = _settings("sync", steps=4, execution_steps=4)
    executor = ChunkExecutor(settings.smoothing, np.full(7, 0.25))
    obs_fn = deploy.make_obs_fn(robot, settings.spec, settings.layout, "pick", None)
    assert deploy.run_episode(robot, LeftOnly(8), None, settings, executor, obs_fn, None, _args()) == 4
    assert len(robot.sent_arm) == 4 and robot.sent_arm[-1][7:].tolist() == [0.25] * 7
    assert len(robot.sent_hands["left"]) == 4 and robot.sent_hands["right"] == []


def test_async_episode_executes_cached_chunks(no_pacing) -> None:
    robot = FakeRobot(frame_hw=(8, 8))
    policy = FakePolicy(horizon=8, latency=0.002)
    obs_fn = deploy.make_obs_fn(
        robot, observation.default_spec(tactile=False), observation.CameraLayout(), "pick", None
    )
    worker = AsyncPolicyWorker(policy, obs_fn, log_every=0)
    worker.start()
    worker.resume()
    try:
        executed = _run(robot, policy, worker, _settings("async", steps=8, execution_steps=4))
    finally:
        worker.stop()
    assert executed == 8 and len(robot.sent_arm) == 8
    versions = []
    for i, row in enumerate(robot.sent_arm):
        k, t = divmod(float(row[0]), 100.0)
        assert t == i % 4, "start_step is 0 for a fresh chunk at dt=10 s"
        assert row[7] == row[0] + 0.5
        versions.append(int(k))
    assert versions[:4] == [versions[0]] * 4 and versions[4:] == [versions[4]] * 4
    assert versions[4] > versions[0], "the second chunk is a newer worker version"
    assert policy.calls >= 2


def test_same_plan_episode_drives_slow_and_refine_in_order(no_pacing) -> None:
    robot = FakeRobot(frame_hw=(8, 8))
    policy = FakePolicy(horizon=16)
    worker = TRexPlanWorker(policy)
    worker.start()
    try:
        executed = _run(robot, policy, worker, _settings("same-plan", steps=16, execution_steps=8))
        # The slow pass queued at the last boundary may still be in flight.
        deadline = threading.Event()
        for _ in range(200):
            if len(policy.options) >= 5:
                break
            deadline.wait(0.005)
    finally:
        worker.stop()
    assert executed == 16 and len(robot.sent_arm) == 16
    flags = [options["trex_force_slow"] for options in policy.options]
    assert flags[:5] == [True, False, True, False, True], "slow, refine @t=4, slow (lead 3), refine, slow"
    assert all(options["trex_manual"] is True for options in policy.options)
    for i, row in enumerate(robot.sent_arm):
        assert float(row[0]) % 100.0 == i % 8, "plan index survives the same-plan splice"


def test_same_plan_contact_gate_skips_refines_without_contact(no_pacing) -> None:
    robot = FakeRobot(frame_hw=(8, 8))
    policy = FakePolicy(horizon=16)
    worker = TRexPlanWorker(policy)
    worker.start()
    try:
        _run(robot, policy, worker, _settings("same-plan", steps=8, execution_steps=8), _args("--trex-contact-gate"))
    finally:
        worker.stop()
    assert [options["trex_force_slow"] for options in policy.options] == [True] * len(policy.options)
    assert len(policy.options) >= 1


# ── run(): startup, episode framing, teardown ────────────────────────────────


class FakeServer(FakePolicy):
    """``GR00TClient`` stand-in: the policy plus ping / server_info / close."""

    def __init__(self, info, horizon: int = 8) -> None:
        super().__init__(horizon=horizon, latency=0.001)
        self.info = info
        self.pings = 0
        self.closed = False

    def ping(self) -> bool:
        self.pings += 1
        return True

    def server_info(self):
        return self.info

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def wired(monkeypatch, no_pacing):
    """``run()`` against FakeRobot and FakeServer, Enter pressed at the prompt, labels answered ``success``."""
    robots: list[FakeRobot] = []
    real_make_robot = deploy.make_robot

    def make_robot(args):
        robot = real_make_robot(args)
        robots.append(robot)
        return robot

    server = FakeServer(_info())
    monkeypatch.setattr(deploy, "make_robot", make_robot)
    monkeypatch.setattr(deploy, "GR00TClient", lambda host, port, timeout_ms: server)
    monkeypatch.setattr(deploy, "prompt_episode_label", lambda: "success")
    monkeypatch.setattr("builtins.input", lambda prompt="": "")
    return server, robots


BASE_RUN = [
    "--fake-robot",
    "--no-tactile",
    "--steps",
    "8",
    "--action-horizon",
    "4",
    "--frequency",
    "0.1",
    "--hold-s",
    "0",
]


def test_run_sync_frames_one_episode(wired, tmp_path: Path) -> None:
    server, robots = wired
    log = tmp_path / "results.jsonl"
    code = deploy.main(BASE_RUN + ["--num-episodes", "1", "--init-state-hold-s", "0", "--log-results", str(log)])
    assert code == 0
    (robot,) = robots
    assert server.pings >= 1 and server.resets == 1 and server.calls == 2
    assert len(robot.sent_arm) == 8 and len(robot.sent_hands["left"]) == 8
    # Home at startup, after the episode and at teardown. Without a YAML the startup init staging is a no-op, but
    # the right-arm hold measured after it makes the post-episode staging run (left zeros, right = hold), as the
    # old client did.
    assert robot.home_calls == 3 and robot.init_calls == 1
    assert robot.closed and server.closed
    record = json.loads(log.read_text().splitlines()[0])
    assert record["label"] == "success" and record["tactile"] is False and record["steps"] == 8


def test_run_async_stops_its_worker(wired) -> None:
    server, robots = wired
    code = deploy.main(BASE_RUN + ["--num-episodes", "1", "--async-policy"])
    assert code == 0
    (robot,) = robots
    assert len(robot.sent_arm) == 8 and server.resets == 1 and server.calls >= 2
    assert all(options is None for options in server.options)
    assert not any(thread.name == "async-policy-worker" and thread.is_alive() for thread in threading.enumerate())
    assert robot.closed and server.closed


def test_run_quit_at_the_prompt_skips_inference(wired, monkeypatch) -> None:
    server, robots = wired
    monkeypatch.setattr("builtins.input", lambda prompt="": "q")
    assert deploy.main(BASE_RUN) == 0
    (robot,) = robots
    assert server.calls == 0 and server.resets == 0
    assert robot.home_calls == 1 and robot.sent_arm == [] and robot.closed and server.closed


def test_run_init_state_and_right_arm_hold(wired, tmp_path: Path) -> None:
    server, robots = wired
    yaml_path = tmp_path / "init.yaml"
    yaml_path.write_text(
        "states:\n  left_arm:\n    qpos: [0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]\n"
        "  right_arm:\n    qpos: [0.2, 0.2, 0.2, 0.2, 0.2, 0.2, 0.2]\n"
        f"  left_ee:\n    qpos: {[0.3] * 22}\n"
    )
    code = deploy.main(
        BASE_RUN + ["--num-episodes", "1", "--init-state-yaml", str(yaml_path), "--init-state-hold-s", "0"]
    )
    assert code == 0
    (robot,) = robots
    assert robot.init_calls == 2, "init posture staged at startup and after the episode"
    assert robot.home_calls == 3 and len(robot.sent_arm) == 8

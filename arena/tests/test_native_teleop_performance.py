# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the pinned native tool without booting Kit or requiring input hardware."""

from __future__ import annotations

import ast
import contextlib
import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import torch

ROOT = Path(__file__).resolve().parents[2]


def functions(path, names, namespace):
    tree = ast.parse(path.read_text())
    nodes = [node for node in tree.body if isinstance(node, (ast.FunctionDef, ast.ClassDef)) and node.name in names]
    exec(
        compile(
            ast.Module(body=nodes, type_ignores=[]),
            str(path),
            "exec",
            flags=__import__("__future__").annotations.compiler_flag,
        ),
        namespace,
    )
    return namespace


def tool_namespace(no_record):
    args = SimpleNamespace(
        no_record=no_record, num_success_steps=2, num_demos=1, xr=False, reset_sim_buffer_each_episode=False
    )
    return functions(
        ROOT / "third_party/IsaacLab-3939445/scripts/tools/record_demos.py",
        {"setup_output_directories", "process_success_condition", "handle_reset", "run_simulation_loop"},
        dict(args_cli=args, torch=torch, contextlib=contextlib, time=SimpleNamespace(time=lambda: 0)),
    )


@pytest.mark.parametrize("record", [False, True])
def test_success_exports_only_when_recording(record):
    ns = tool_namespace(not record)
    env = SimpleNamespace(recorder_manager=Mock(), device="cpu")
    term = SimpleNamespace(func=lambda env: [True], params={})
    count, done = ns["process_success_condition"](env, term, 0)
    assert (count, done) == (1, False)
    assert ns["process_success_condition"](env, term, count) == (2, True)
    assert env.recorder_manager.export_episodes.call_count == int(record)


def test_no_capture_does_not_create_output_directories():
    assert tool_namespace(True)["setup_output_directories"]() == ("", "")


@pytest.mark.parametrize("capture_stack", [False, True])
@pytest.mark.parametrize("record", [False, True])
def test_native_and_xr_pause_reset_episode_limit_and_cleanup(monkeypatch, capture_stack, record):
    ns = tool_namespace(not record)
    ns["args_cli"].xr = capture_stack
    # Skip the historical exit delay; it has no bearing on state transitions.
    clock = iter(range(100))
    ns["time"] = SimpleNamespace(time=lambda: next(clock))
    callbacks = {}
    haptic = SimpleNamespace(update=Mock(), stop=Mock())
    events = iter(
        [
            SimpleNamespace(is_active=value, should_reset=reset)
            for value, reset in [(True, False), (False, False), (True, True), (True, False), (True, False)]
        ]
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_teleop",
        SimpleNamespace(
            create_haptic_feedback_driver=lambda *a: haptic, poll_control_events=lambda device: next(events)
        ),
    )
    monkeypatch.setitem(
        sys.modules, "isaaclab_mimic.ui.instruction_display", SimpleNamespace(show_subtask_instructions=lambda *a: None)
    )
    checks = iter([False, False, True, True])
    term = SimpleNamespace(func=lambda env: [next(checks)], params={})
    env = SimpleNamespace(
        sim=SimpleNamespace(
            has_gui=False, is_running=lambda: True, is_stopped=lambda: False, render=Mock(), reset=Mock()
        ),
        num_envs=1,
        device="cpu",
        cfg=SimpleNamespace(),
        reset=Mock(),
        step=Mock(return_value=({},)),
        recorder_manager=Mock(),
    )
    env.unwrapped = env

    class Device:
        reset = Mock()
        close = Mock()
        __enter__ = Mock(return_value=None)
        __exit__ = Mock(return_value=None)
        calls = 0

        def advance(self):
            self.calls += 1
            if not capture_stack:
                if self.calls == 2:
                    callbacks["STOP"]()
                if self.calls == 3:
                    callbacks["START"]()
                    callbacks["RESET"]()
            return torch.zeros(4)

    device = Device()

    def setup(cb, stack):
        callbacks.update(cb)
        return device

    ns.update(setup_teleop_device=setup, setup_ui=lambda *a: SimpleNamespace(show_demo=Mock()), env_cfg=env.cfg)
    camera = SimpleNamespace(bind=lambda env: contextlib.nullcontext(), refresh=Mock())
    assert ns["run_simulation_loop"](env, None, term, None, camera, capture_stack) == 1
    assert env.step.call_count == 4  # paused iteration does not advance success or physics
    assert env.reset.call_count == 2
    assert not env.sim.reset.called  # navigation keeps its existing reset flag
    assert camera.refresh.call_count == 1
    assert env.recorder_manager.export_episodes.call_count == int(record)
    assert device.__exit__.called if capture_stack else device.close.called
    if capture_stack:
        assert haptic.stop.called


@pytest.mark.parametrize(
    "requested,reason,expected",
    [("auto", None, "cuda"), ("auto", "missing NVRTC", "vulkan"), ("vulkan", None, "vulkan")],
)
def test_bootstrap_selects_and_imports_before_native_runner(monkeypatch, requested, reason, expected):
    import i4h_arena.medical.slang_cuda as cuda

    spec = importlib.util.spec_from_file_location("bootstrap", ROOT / "scripts/native_fluoro_bootstrap.py")
    bootstrap = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(bootstrap)
    order = []
    monkeypatch.setattr(cuda, "check_slang_cuda_device", lambda: reason)
    monkeypatch.setattr(cuda, "prepare_slang_for_kit", lambda device: order.append(("prepare", device)))
    monkeypatch.setenv("I4H_FLUORO_DEVICE_REQUEST", requested)
    monkeypatch.delenv("I4H_RECORDING_METADATA", raising=False)
    monkeypatch.setattr(sys, "argv", ["bootstrap", "-m", "native.runner", "--arg"])
    monkeypatch.setattr(bootstrap.runpy, "run_module", lambda *a, **k: order.append(("runner", sys.argv.copy())))
    bootstrap.main()
    assert order == [("prepare", expected), ("runner", ["native.runner", "--arg", "--fluoro_device", expected])]


def test_explicit_cuda_failure_is_actionable(monkeypatch):
    import i4h_arena.medical.slang_cuda as cuda

    ns = functions(ROOT / "scripts/native_fluoro_bootstrap.py", {"resolve_device"}, {})
    monkeypatch.setattr(cuda, "check_slang_cuda_device", lambda: "missing NVRTC")
    with pytest.raises(RuntimeError, match="missing NVRTC.*--fluoro-device vulkan"):
        ns["resolve_device"]("cuda")


def test_empty_sensor_mask_skips_even_provider_and_renderer_initialization():
    ns = functions(
        ROOT / "arena/i4h_arena/sensors/fluoroscopy.py",
        {"FluoroscopySensor"},
        dict(SensorBase=object, wp=SimpleNamespace(to_torch=lambda x: x), torch=torch),
    )
    sensor = object.__new__(ns["FluoroscopySensor"])
    sensor._device = "cpu"
    # Deliberately no data, providers or renderer: any access would fail.
    sensor._update_buffers_impl(torch.zeros(3, dtype=torch.bool))


@pytest.mark.parametrize("record", [False, True])
def test_config_disables_all_dataset_terms_and_keeps_progress_hook(record, monkeypatch):
    from isaaclab.managers import CommandTermCfg, DatasetExportMode
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg

    cfg = SimpleNamespace(
        sim=SimpleNamespace(),
        scene=SimpleNamespace(),
        terminations=SimpleNamespace(time_out=object()),
        commands=None,
        observations=SimpleNamespace(policy=SimpleNamespace(concatenate_terms=True)),
    )
    ns = tool_namespace(not record)
    ns["args_cli"].teleop_device = "leader"
    ns["args_cli"].device = "cpu"
    ns["args_cli"].task = "test"
    ns.update(
        resolve_task_config=lambda *a: (cfg, None),
        _prepare_success_term=lambda cfg: object(),
        logger=Mock(),
        CommandTermCfg=CommandTermCfg,
        DatasetExportMode=DatasetExportMode,
        gym=SimpleNamespace(spec=lambda *a: SimpleNamespace(kwargs={})),
        ActionStateRecorderManagerCfg=RecorderManagerBaseCfg,
    )
    functions(ROOT / "third_party/IsaacLab-3939445/scripts/tools/record_demos.py", {"create_environment_config"}, ns)
    result, term, xr = ns["create_environment_config"]("/tmp/unused", "capture")
    assert term is not None and xr is False
    if record:
        assert result.recorders.dataset_export_mode == DatasetExportMode.EXPORT_SUCCEEDED_ONLY
        assert result.recorders.dataset_filename == "capture"
    else:
        assert result.recorders.dataset_export_mode == DatasetExportMode.EXPORT_NONE
        assert result.recorders.recording_enabled is False
        assert hasattr(result.recorders, "progress_hook")
        assert [k for k in vars(result.recorders) if k.startswith("record_")] == []


def test_partial_sensor_mask_preserves_unselected_buffers_and_frames():
    from i4h_arena.medical.catheter import CatheterState

    class Renderer:
        def render(self, *args, env_mask):
            assert env_mask.tolist() == [False, True, False]
            return {"rgb": torch.full((3, 2, 2, 3), 255, dtype=torch.uint8)}

    ns = functions(
        ROOT / "arena/i4h_arena/sensors/fluoroscopy.py",
        {"FluoroscopySensor"},
        dict(
            SensorBase=object,
            wp=SimpleNamespace(to_torch=lambda x: x),
            torch=torch,
            CatheterState=CatheterState,
            SlangFluoroscopyRenderer=Renderer,
        ),
    )
    sensor = object.__new__(ns["FluoroscopySensor"])
    sensor._device = "cpu"
    sensor._num_envs = 3
    sensor.display_revision = __import__("numpy").array([4, 5, 6])
    sensor._data = SimpleNamespace(
        frame_id=torch.tensor([4, 5, 6]), output={"rgb": torch.zeros((3, 2, 2, 3), dtype=torch.uint8)}
    )
    sensor._catheter_provider = None
    sensor._carm_provider = SimpleNamespace(snapshot=lambda n: object())
    sensor.cfg = SimpleNamespace(backend="slang")
    sensor._renderer = Renderer()
    sensor._ensure_renderer = lambda carm: None
    sensor._update_buffers_impl(torch.tensor([False, True, False]))
    assert sensor._data.frame_id.tolist() == [4, 6, 6]
    assert sensor.display_revision.tolist() == [4, 6, 6]
    assert sensor._data.output["rgb"][1].eq(255).all()
    assert sensor._data.output["rgb"][[0, 2]].eq(0).all()


@pytest.mark.parametrize("record", [False, True])
def test_recorder_callbacks_can_track_progress_without_growing_episode_buffers(record):
    from isaaclab.managers import DatasetExportMode, RecorderManager, RecorderTerm, RecorderTermCfg
    from isaaclab.managers.recorder_manager import RecorderManagerBaseCfg
    from isaaclab.utils import configclass

    calls = []

    class Progress(RecorderTerm):
        def record_post_step(self):
            calls.append("progress")
            return "progress", torch.ones((1, 1))

    @configclass
    class Cfg(RecorderManagerBaseCfg):
        dataset_export_mode = DatasetExportMode.EXPORT_NONE
        recording_enabled = record
        progress = RecorderTermCfg(class_type=Progress)

    env = SimpleNamespace(
        num_envs=1,
        device="cpu",
        sim=SimpleNamespace(is_playing=lambda: True),
        cfg=SimpleNamespace(
            sim=SimpleNamespace(dt=1 / 120, render_interval=4), decimation=4, scene=SimpleNamespace(num_envs=1)
        ),
    )
    manager = RecorderManager(Cfg(), env)
    try:
        for _ in range(100):
            manager.record_post_step()
        assert len(calls) == 100
        assert manager.get_episode(0).is_empty() == (not record)
        assert manager._dataset_file_handler is None
    finally:
        manager.close()


@pytest.mark.parametrize("device", ["so101_leader", "keyboard", "xr"])
def test_native_device_factories_preserve_callbacks_and_xr_haptics(device, monkeypatch):
    callbacks = {"START": Mock(), "STOP": Mock(), "RESET": Mock()}
    factory = Mock(return_value=object())
    ns = tool_namespace(True)
    args = ns["args_cli"]
    args.teleop_device = None if device == "xr" else device
    args.device = "cpu"
    args.cloudxr_env = None
    args.auto_launch_cloudxr = False
    args.mcap_record_path = None
    args.enable_debug_visualization = False
    haptics = object()
    cfg = SimpleNamespace(
        teleop_devices=SimpleNamespace(devices={device: object()}), isaac_teleop=object(), haptic_feedback=haptics
    )
    monkeypatch.setitem(sys.modules, "isaaclab_teleop", SimpleNamespace(create_isaac_teleop_device=factory))
    ns.update(env_cfg=cfg, create_teleop_device=factory, logger=Mock(), _resolve_cloudxr_env=lambda *a: None)
    functions(ROOT / "third_party/IsaacLab-3939445/scripts/tools/record_demos.py", {"setup_teleop_device"}, ns)
    assert ns["setup_teleop_device"](callbacks, device == "xr") is factory.return_value
    if device == "xr":
        assert factory.call_args.kwargs["callbacks"] is callbacks
        assert factory.call_args.kwargs["haptic_cfg"] is haptics
    else:
        assert factory.call_args.args == (device, cfg.teleop_devices.devices, callbacks)


@pytest.mark.parametrize("capture_stack", [False, True])
@pytest.mark.parametrize("action", [None, torch.zeros(4)])
def test_missing_actions_and_paused_iterations_share_limiter(monkeypatch, capture_stack, action):
    ns = tool_namespace(True)
    ns["args_cli"].xr = capture_stack
    ns["args_cli"].num_demos = 0
    monkeypatch.setitem(
        sys.modules, "isaaclab_mimic.ui.instruction_display", SimpleNamespace(show_subtask_instructions=lambda *a: None)
    )
    monkeypatch.setitem(
        sys.modules,
        "isaaclab_teleop",
        SimpleNamespace(
            create_haptic_feedback_driver=lambda *a: None,
            poll_control_events=lambda device: SimpleNamespace(is_active=False, should_reset=False),
        ),
    )
    running = iter([True, True, True, False])
    env = SimpleNamespace(
        sim=SimpleNamespace(has_gui=False, is_running=lambda: next(running), is_stopped=lambda: False, render=Mock()),
        num_envs=1,
        cfg=SimpleNamespace(),
        step=Mock(),
        reset=Mock(),
    )
    env.unwrapped = env

    class Device:
        def reset(self):
            pass

        def advance(self):
            return action

        def __enter__(self):
            return self

        def __exit__(self, *args):
            pass

        def close(self):
            pass

    def setup(callbacks, stack):
        callbacks["STOP"]()
        return Device()

    ns.update(setup_teleop_device=setup, setup_ui=lambda *a: SimpleNamespace(show_demo=Mock()), env_cfg=env.cfg)
    camera = SimpleNamespace(bind=lambda env: contextlib.nullcontext())
    pace = Mock()
    ns["run_simulation_loop"](env, None, None, pace, camera, capture_stack)
    assert pace.begin.call_count == pace.sleep.call_count == 3
    assert not env.step.called
    assert not env.sim.render.called  # the shared limiter owns the one viewport pump


@pytest.mark.parametrize("selected", [None, [1], slice(0, 2), torch.tensor([2])])
def test_sensor_reset_invalidates_only_selected_display_environments(selected):
    import numpy as np

    class Base:
        def reset(self, **kwargs):
            self.reset_args = kwargs

    ns = functions(
        ROOT / "arena/i4h_arena/sensors/fluoroscopy.py",
        {"FluoroscopySensor"},
        dict(SensorBase=Base, np=np, torch=torch),
    )
    sensor = object.__new__(ns["FluoroscopySensor"])
    sensor.display_revision = np.array([4, 6, 8])
    expected = sensor.display_revision.copy()
    expected[slice(None) if selected is None else selected] += 1
    sensor.reset(env_ids=selected)
    np.testing.assert_array_equal(sensor.display_revision, expected)


def test_sensor_reset_mask_has_priority_and_preserves_peer_revisions():
    import numpy as np

    class Base:
        def reset(self, **kwargs):
            pass

    ns = functions(
        ROOT / "arena/i4h_arena/sensors/fluoroscopy.py",
        {"FluoroscopySensor"},
        dict(SensorBase=Base, np=np, torch=torch, wp=SimpleNamespace(to_torch=lambda value: value)),
    )
    sensor = object.__new__(ns["FluoroscopySensor"])
    sensor.display_revision = np.array([4, 6, 8])
    sensor.reset(env_ids=[0, 2], env_mask=torch.tensor([False, True, False]))
    np.testing.assert_array_equal(sensor.display_revision, [4, 7, 8])

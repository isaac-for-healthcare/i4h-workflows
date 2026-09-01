# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive as a workflow node, exercised with a fake device and an in-process bus."""

from __future__ import annotations

import sys
import types

import numpy as np
import pytest

from i4h_common.bus.inproc import InProcBus
from i4h_common.bus.messages import RobotCommand, encode
from i4h_common.paths import workflow_root
from i4h_engine.discover import discover_tasks
from i4h_engine.executor import Engine
from i4h_engine.graph import TaskGraph, node
from i4h_engine.status import Status, WorkflowStatus
from i4h_engine.task import TickContext
from i4h_tasks.basic.testing.fake_scene import FakeActuation, FakeScene
from i4h_tasks.teleop.devices import (
    BusDevice,
    CatheterKeyboardDevice,
    InputDevice,
    KeyboardDevice,
    keyboard_event_input_name,
    make_device,
)
from i4h_tasks.teleop.drive import Drive

DT = 1 / 60
SPECS = {
    task_id: spec.resolve() for task_id, spec in discover_tasks(workflow_root())[0].items() if spec.project == "teleop"
}


class ScriptedDevice(InputDevice):
    """Replays a canned sequence, then reports done."""

    def __init__(self, frames: list[np.ndarray] | None = None, *, gaps: bool = False) -> None:
        self.frames = frames or []
        self.gaps = gaps
        self.index = 0
        self.opened = False
        self.closed = False

    def open(self, ctx: TickContext) -> None:
        self.opened = True

    def read(self, ctx: TickContext) -> np.ndarray | None:
        if self.gaps and self.index % 2 == 0:
            self.index += 1
            return None
        if self.index >= len(self.frames):
            return None
        frame = self.frames[self.index]
        self.index += 1
        return np.tile(frame, (ctx.num_envs, 1))

    @property
    def done(self) -> bool:
        return self.index >= len(self.frames) and not self.gaps

    def close(self) -> None:
        self.closed = True


@pytest.fixture
def ctx():
    return TickContext(scene=FakeScene(dof=6), act=FakeActuation(dof=6), dt=DT)


def _drive_with(device: InputDevice, **kwargs) -> Drive:
    task = Drive(name="drive", **kwargs)
    task._device_override = device  # noqa: SLF001
    original = task.on_enter

    def on_enter(ctx, inputs):  # noqa: ANN001
        task._device = device  # noqa: SLF001
        device.open(ctx)
        task._frames = 0  # noqa: SLF001
        task._ticks = 0  # noqa: SLF001
        task._completed = False  # noqa: SLF001

    task.on_enter = on_enter  # type: ignore[method-assign]
    assert original is not None
    return task


# -- device resolution ---------------------------------------------------


def test_make_device_known_names():
    assert make_device("keyboard") is not None
    assert isinstance(make_device("vr"), BusDevice)
    assert isinstance(make_device("bus"), BusDevice)


def test_make_device_unknown_name():
    with pytest.raises(KeyError, match="unknown teleop device"):
        make_device("mind_control")


def test_make_device_ignores_irrelevant_kwargs():
    # run.sh passes every teleop flag; a device must take only what it knows.
    assert make_device("keyboard", sensitivity=2.0, port="/dev/ttyACM9") is not None


def _fake_isaac_keyboard(monkeypatch, command):
    class Config:
        def __init__(self, **kwargs):
            self.__dict__.update(kwargs)

    class Keyboard:
        def __init__(self, cfg):
            self.cfg = cfg

        def advance(self):
            return np.asarray(command, dtype=np.float32)

    isaaclab = types.ModuleType("isaaclab")
    devices = types.ModuleType("isaaclab.devices")
    devices.Se3Keyboard = Keyboard
    devices.Se3KeyboardCfg = Config
    isaaclab.devices = devices
    monkeypatch.setitem(sys.modules, "isaaclab", isaaclab)
    monkeypatch.setitem(sys.modules, "isaaclab.devices", devices)


def test_keyboard_passes_relative_cartesian_delta(monkeypatch):
    _fake_isaac_keyboard(monkeypatch, [0.01, -0.02, 0.03, 0.04, -0.05, 0.06])
    local = TickContext(
        scene=FakeScene(dof=7),
        act=FakeActuation(dof=6, action_space="ee_pose"),
        dt=DT,
    )
    device = KeyboardDevice()
    device.open(local)
    assert device._impl.cfg.gripper_term is False  # noqa: SLF001
    assert np.allclose(device.read(local), [[0.01, -0.02, 0.03, 0.04, -0.05, 0.06]])


def test_keyboard_maps_joint_arm_and_gripper(monkeypatch):
    _fake_isaac_keyboard(monkeypatch, [1.0, 2.0, 3.0, 4.0, 5.0, 6.0, -1.0])
    device = KeyboardDevice(step_rad=0.02)
    device.open(ctx := TickContext(scene=FakeScene(dof=6), act=FakeActuation(dof=6), dt=DT))
    assert device._impl.cfg.gripper_term is True  # noqa: SLF001
    assert np.allclose(device.read(ctx), [[0.02, 0.04, 0.06, 0.08, 0.10, -0.16]])


def test_catheter_keyboard_maps_insertion_rotation_and_orbit() -> None:
    local = TickContext(
        scene=FakeScene(dof=3, joint_names=("insertion_m", "rotation_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=3, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(insertion_speed_mps=0.012, rotation_rate_radps=0.8, orbit_rate_radps=0.45)
    device._keyboard_sub = object()  # noqa: SLF001
    device._pressed.update(("W", "A", "Q"))  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.012, -0.8, 0.45]])


def test_catheter_keyboard_drives_named_carm_projection() -> None:
    scene = FakeScene(dof=3, joint_names=("insertion_m", "rotation_rad", "carm_orbit_rad"))
    local = TickContext(
        scene=scene,
        act=FakeActuation(dof=3, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(orbit_rate_radps=0.45)
    device._keyboard_sub = object()  # noqa: SLF001
    device._orbit_target_rad = np.deg2rad(45.0)  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.45]])


def test_catheter_keyboard_uses_live_velocity_control() -> None:
    local = TickContext(
        scene=FakeScene(dof=3, joint_names=("insertion_m", "rotation_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=3, action_space="catheter_carm_velocity"),
        dt=DT,
        controls={"catheter_insertion_speed_mps": 0.027},
    )
    device = CatheterKeyboardDevice()
    device._keyboard_sub = object()  # noqa: SLF001
    device._pressed.add("W")  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.027, 0.0, 0.0]])


def test_catheter_keyboard_requests_full_scene_reset() -> None:
    local = TickContext(
        scene=FakeScene(dof=3, joint_names=("insertion_m", "rotation_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=3, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice()
    device._keyboard_sub = object()  # noqa: SLF001
    device._reset_requested = True  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.0]])
    assert local.consume_scene_reset() is True
    assert local.consume_scene_reset() is False


@pytest.mark.parametrize(
    "value",
    ["W", "w", "KeyboardInput.W", "KEY_W", types.SimpleNamespace(name="W")],
)
def test_keyboard_event_input_name_accepts_string_and_named_input(value) -> None:
    assert keyboard_event_input_name(types.SimpleNamespace(input=value)) == "W"


def test_catheter_keyboard_rejects_an_incompatible_action_space():
    device = CatheterKeyboardDevice()

    with pytest.raises(RuntimeError, match="catheter_carm_velocity"):
        device.open(TickContext(scene=FakeScene(dof=6), act=FakeActuation(dof=6), dt=DT))


# -- bus device ----------------------------------------------------------


def test_bus_device_reads_commands(ctx):
    bus = InProcBus()
    ctx.bus = bus
    device = BusDevice("i4h/p/robot/command")
    device.open(ctx)
    assert device.read(ctx) is None
    bus.publish("i4h/p/robot/command", encode(RobotCommand(joint_positions=[0.1] * 6)))
    command = device.read(ctx)
    assert command is not None and np.allclose(command, 0.1)
    # take() semantics: a stale command must not be re-applied forever.
    assert device.read(ctx) is None
    device.close()


def test_bus_device_needs_a_bus(ctx):
    ctx.bus = None
    with pytest.raises(RuntimeError, match="needs a bus"):
        BusDevice("k").open(ctx)


def test_bus_device_ignores_empty_command(ctx):
    bus = InProcBus()
    ctx.bus = bus
    device = BusDevice("k")
    device.open(ctx)
    bus.publish("k", encode(RobotCommand(joint_positions=[])))
    assert device.read(ctx) is None


# -- the task ------------------------------------------------------------


def test_teleop_applies_device_frames(ctx):
    device = ScriptedDevice([np.full(6, 0.2, dtype=np.float32)] * 3)
    task = _drive_with(device)
    task.on_enter(ctx, Drive.Inputs())
    assert task.tick(ctx) is Status.RUNNING
    assert np.allclose(ctx.act.raw_actions["robot"], 0.2)
    assert device.opened


def test_teleop_holds_when_the_device_has_nothing(ctx):
    # A polled device returning None is normal, not an error; holding keeps the
    # arm where the human left it instead of snapping to zero.
    device = ScriptedDevice([np.zeros(6, np.float32)] * 4, gaps=True)
    task = _drive_with(device)
    task.on_enter(ctx, Drive.Inputs())
    task.tick(ctx)
    assert ctx.act.holds == ["robot"]


def test_teleop_finishes_when_operator_signals_done(ctx):
    device = ScriptedDevice([np.zeros(6, np.float32)])
    task = _drive_with(device)
    task.on_enter(ctx, Drive.Inputs())
    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.SUCCESS
    out = task.on_exit(ctx)
    assert out.completed is True
    assert out.frames == 1
    assert device.closed


def test_teleop_finishes_on_predicate(ctx):
    calls = {"n": 0}

    def until(_c):
        calls["n"] += 1
        return calls["n"] >= 2

    device = ScriptedDevice([np.zeros(6, np.float32)] * 100)
    task = _drive_with(device, until=until)
    task.on_enter(ctx, Drive.Inputs())
    assert task.tick(ctx) is Status.RUNNING
    assert task.tick(ctx) is Status.SUCCESS


def test_teleop_fails_on_budget(ctx):
    device = ScriptedDevice([np.zeros(6, np.float32)] * 10_000)
    task = _drive_with(device, max_seconds=0.03)
    task.on_enter(ctx, Drive.Inputs())
    statuses = [task.tick(ctx) for _ in range(5)]
    assert statuses[-1] is Status.FAILURE


def test_teleop_releases_the_device_on_abort(ctx):
    device = ScriptedDevice([np.zeros(6, np.float32)] * 10)
    task = _drive_with(device)
    task.on_enter(ctx, Drive.Inputs())
    task.tick(ctx)
    task.on_abort(ctx)
    assert device.closed


def test_teleop_runs_inside_a_workflow(ctx):
    device = ScriptedDevice([np.full(6, 0.3, np.float32)] * 2)
    task = _drive_with(device)
    engine = Engine(TaskGraph().flow(node(task)))
    engine.start(ctx)
    for _ in range(20):
        if engine.status.is_terminal:
            break
        engine.tick(ctx)
    assert engine.status is WorkflowStatus.SUCCEEDED


# -- manifest drift ------------------------------------------------------

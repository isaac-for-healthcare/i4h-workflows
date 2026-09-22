# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Drive as a workflow node, exercised with a fake device and an in-process bus."""

from __future__ import annotations

import sys
import time
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
    key_log_enabled,
    keyboard_event_input_name,
    make_device,
)
from i4h_tasks.teleop.devices import KEY_LOG_ENV_VAR
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
    """Orbit stays last now that the tip bend sits between it and rotation."""
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(insertion_speed_mps=0.012, rotation_rate_radps=0.8, orbit_rate_radps=0.45)
    device._keyboard_sub = object()  # noqa: SLF001
    for held in ("W", "A", "Q"):
        device._mark_held(held)  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.012, -0.8, 0.0, 0.45]])


def test_catheter_keyboard_drives_named_carm_projection() -> None:
    scene = FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad"))
    local = TickContext(
        scene=scene,
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(orbit_rate_radps=0.45)
    device._keyboard_sub = object()  # noqa: SLF001
    device._orbit_target_rad = np.deg2rad(45.0)  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.0, 0.45]])


def test_catheter_keyboard_uses_live_velocity_control() -> None:
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
        controls={"catheter_insertion_speed_mps": 0.027},
    )
    device = CatheterKeyboardDevice()
    device._keyboard_sub = object()  # noqa: SLF001
    device._mark_held("W")  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.027, 0.0, 0.0, 0.0]])


def test_a_key_with_no_fresh_evidence_stops_commanding() -> None:
    """The stuck-key bug, now bounded by the hold window.

    A held key streams ``KEY_REPEAT`` and stops the moment it is let go, so the
    absence of repeats is what ends the command. Releases are not consulted,
    because Kit delivers them unreliably -- one session logged 36 seconds of
    unrequested retraction from a single dropped release, ending only when the
    insertion depth hit its lower bound.
    """
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(insertion_speed_mps=0.030, key_hold_ttl_s=0.05)
    device._keyboard_sub = object()  # noqa: SLF001
    device._mark_held("S")  # noqa: SLF001

    assert np.allclose(device.read(local), [[-0.030, 0.0, 0.0, 0.0]])

    time.sleep(0.2)  # four times the hold window, with no repeat arriving

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.0, 0.0]])


def test_clearing_the_keys_stops_a_stuck_command() -> None:
    """What ``L`` does, for an operator who does not want to wait out the window."""
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(insertion_speed_mps=0.030, key_hold_ttl_s=10.0)
    device._keyboard_sub = object()  # noqa: SLF001
    device._mark_held("S")  # noqa: SLF001
    device._held_since.clear()  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.0, 0.0]])


def test_catheter_keyboard_steers_the_tip_on_z_and_c() -> None:
    """C bends the tip one way, Z the other, without touching the other columns."""
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(tip_bend_rate_radps=1.2)
    device._keyboard_sub = object()  # noqa: SLF001

    device._mark_held("C")  # noqa: SLF001
    assert np.allclose(device.read(local), [[0.0, 0.0, 1.2, 0.0]])

    device._held_since.clear()  # noqa: SLF001
    device._mark_held("Z")  # noqa: SLF001
    assert np.allclose(device.read(local), [[0.0, 0.0, -1.2, 0.0]])


def test_holding_both_tip_keys_cancels_out() -> None:
    """Opposed keys are a zero command rather than whichever arrived last."""
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice(tip_bend_rate_radps=1.2)
    device._keyboard_sub = object()  # noqa: SLF001
    device._mark_held("Z")  # noqa: SLF001
    device._mark_held("C")  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.0, 0.0]])


def test_catheter_keyboard_rejects_the_old_three_value_action_space() -> None:
    """The bend column widened the space, so a stale 3-dof scene must not open.

    Silently accepting it would map orbit onto the bend and steer the tip
    whenever the operator asked for a C-arm sweep.
    """
    device = CatheterKeyboardDevice()
    stale = TickContext(
        scene=FakeScene(dof=3, joint_names=("insertion_m", "rotation_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=3, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    with pytest.raises(RuntimeError, match="four-value"):
        device.open(stale)


def test_the_key_log_is_off_unless_asked_for() -> None:
    assert key_log_enabled({}) is False


@pytest.mark.parametrize("value", ["1", "yes", "true", "on", "please"])
def test_the_key_log_accepts_an_operator_in_a_hurry(value: str) -> None:
    assert key_log_enabled({KEY_LOG_ENV_VAR: value}) is True


@pytest.mark.parametrize("value", ["", "0", "false", "no", "off", "  OFF  "])
def test_the_key_log_stays_off_for_a_negative_setting(value: str) -> None:
    assert key_log_enabled({KEY_LOG_ENV_VAR: value}) is False


def test_catheter_keyboard_defaults_below_the_buckling_speed() -> None:
    """Fast enough to feel direct, slow enough that the shaft can keep up.

    This used to default near the action term's 60 mm/s ceiling, to compensate
    for a fluoroscopy scene that stepped around two hertz -- at that rate a
    slower speed made holding W worth about a millimetre a second, which reads
    as a broken key. Two things retired that reasoning. The slow stepping was
    largely a starved machine, and the scene now runs at 25 Hz. And a hold that
    actually sustains 30 mm/s drove wall penetration from +0.55 to +4.25 mm with
    8 of 41 particles outside the lumen, never recovering, where 9 mm/s held
    penetration negative and every particle contained for 10,000 steps.

    So the ceiling is the wrong thing to track. The speed the shaft can shed is.
    """
    assert CatheterKeyboardDevice().insertion_speed_mps == pytest.approx(0.009)


def test_a_tapped_key_survives_to_the_next_step_on_a_slow_scene() -> None:
    """The hold window is spent in wall time; the steps it feeds are not.

    At roughly two steps a second the old 200 ms window closed before the next
    step ran, so a tap could land entirely between steps and do nothing. The
    window has to outlast one step at that rate to be worth offering at all.
    """
    slowest_expected_step_rate_hz = 2.0

    assert CatheterKeyboardDevice().key_hold_ttl_s >= 1.0 / slowest_expected_step_rate_hz


def test_a_key_still_reads_as_active_within_the_hold_window() -> None:
    """What that window buys: a tap outlives the gap to the next step."""
    device = CatheterKeyboardDevice(key_hold_ttl_s=10.0)
    device._held_since["W"] = time.monotonic()  # noqa: SLF001

    assert device._active("W")  # noqa: SLF001


def test_a_key_last_seen_before_the_window_is_not_active() -> None:
    """And it does expire, so the catheter stops when the operator lets go."""
    device = CatheterKeyboardDevice(key_hold_ttl_s=0.01)
    device._held_since["W"] = time.monotonic() - 1.0  # noqa: SLF001

    assert not device._active("W")  # noqa: SLF001


def test_a_repeat_refreshes_a_hold_that_would_otherwise_expire() -> None:
    """Why repeats are counted: they are the only reliable evidence of a hold.

    Kit sends ``KEY_REPEAT`` continuously while a key is down, and one logged
    session held a key for several seconds without ever producing a ``KEY_PRESS``
    for it. Ignoring repeats, as the handler used to, made that hold invisible.
    """
    device = CatheterKeyboardDevice(key_hold_ttl_s=0.05)
    device._held_since["W"] = time.monotonic() - 1.0  # noqa: SLF001
    assert not device._active("W")  # noqa: SLF001

    device._mark_held("W")  # noqa: SLF001

    assert device._active("W")  # noqa: SLF001


class _FakeInput:
    """The one carb call the device polls, plus a switch to make it fail."""

    def __init__(self, down: set[str], *, raises: bool = False) -> None:
        self.down = down
        self.raises = raises
        self.calls = 0

    def get_keyboard_value(self, _keyboard: object, button: str) -> float:
        self.calls += 1
        if self.raises:
            raise RuntimeError("no input provider")
        return 1.0 if button in self.down else 0.0


def _polling_device(down: set[str], **kwargs) -> CatheterKeyboardDevice:
    device = CatheterKeyboardDevice(**kwargs)
    device._input = _FakeInput(down)  # noqa: SLF001
    device._keyboard = object()  # noqa: SLF001
    device._motion_buttons = {key: key for key in ("W", "S", "A", "D", "Q", "E")}  # noqa: SLF001
    return device


def test_a_held_key_reads_as_held_with_no_events_at_all() -> None:
    """The failure this replaces: Kit sent only ``CHAR`` and the hold went dead."""
    device = _polling_device({"W"})

    assert device._active("W")  # noqa: SLF001
    assert not device._active("S")  # noqa: SLF001


def test_the_device_outranks_a_stale_hold_in_both_directions() -> None:
    """Neither a dropped release nor a dropped press can outlive the real state."""
    device = _polling_device({"W"}, key_hold_ttl_s=10.0)
    device._held_since["S"] = time.monotonic()  # noqa: SLF001

    assert device._active("W")  # noqa: SLF001
    assert not device._active("S")  # noqa: SLF001


def test_polling_failure_falls_back_to_the_events() -> None:
    device = _polling_device({"W"}, key_hold_ttl_s=10.0)
    device._input.raises = True  # noqa: SLF001
    device._mark_held("S")  # noqa: SLF001

    assert not device._active("W")  # noqa: SLF001
    assert device._active("S")  # noqa: SLF001


def test_a_failed_poll_is_not_retried_every_step() -> None:
    """Six keys a step for a whole session is too many raises to swallow."""
    device = _polling_device({"W"})
    device._input.raises = True  # noqa: SLF001

    for _ in range(5):
        device._active("W")  # noqa: SLF001

    assert device._input.calls == 1  # noqa: SLF001


def test_the_events_still_answer_before_the_keyboard_is_opened() -> None:
    device = CatheterKeyboardDevice(key_hold_ttl_s=10.0)
    device._mark_held("W")  # noqa: SLF001

    assert device._active("W")  # noqa: SLF001


def test_catheter_keyboard_requests_full_scene_reset() -> None:
    local = TickContext(
        scene=FakeScene(dof=4, joint_names=("insertion_m", "rotation_rad", "tip_bend_rad", "carm_orbit_rad")),
        act=FakeActuation(dof=4, action_space="catheter_carm_velocity"),
        dt=DT,
    )
    device = CatheterKeyboardDevice()
    device._keyboard_sub = object()  # noqa: SLF001
    device._reset_requested = True  # noqa: SLF001

    assert np.allclose(device.read(local), [[0.0, 0.0, 0.0, 0.0]])
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


# -- a device that opened without attaching ------------------------------


class DetachedDevice(ScriptedDevice):
    """Opens successfully and then never delivers a command.

    What the keyboard devices become when Kit has no window: ``open`` warns
    instead of raising, and every ``read`` returns ``None``.
    """

    @property
    def attached(self) -> bool:
        return False

    @property
    def done(self) -> bool:
        return False

    def read(self, ctx: TickContext) -> np.ndarray | None:
        return None


def _drive_real_on_enter(monkeypatch, device: InputDevice) -> Drive:
    """A ``Drive`` whose real ``on_enter`` runs, unlike :func:`_drive_with`."""
    monkeypatch.setattr("i4h_tasks.teleop.drive.make_device", lambda name, **kwargs: device)
    return Drive(name="drive", device="keyboard")


def test_a_device_that_opened_without_attaching_fails_at_startup(ctx, monkeypatch):
    """The whole point: twenty minutes of driving nothing should not be possible.

    A detached device holds pose on every tick, so the run looks alive, records
    an episode, and reports success. Failing in ``on_enter`` is the only place
    the operator finds out before spending the session.
    """
    task = _drive_real_on_enter(monkeypatch, DetachedDevice([np.zeros(6, np.float32)]))

    with pytest.raises(RuntimeError, match="without acquiring its input source"):
        task.on_enter(ctx, Drive.Inputs())


def test_the_startup_failure_names_the_windowing_cause(ctx, monkeypatch):
    """The operator needs the next step, not just the verdict.

    GLFW's failure is a warning thousands of lines above the symptom, so the
    message that stops the run carries the string to search for and the setting
    to check.
    """
    task = _drive_real_on_enter(monkeypatch, DetachedDevice())

    with pytest.raises(RuntimeError) as failure:
        task.on_enter(ctx, Drive.Inputs())

    assert "GLFW initialization failed" in str(failure.value)
    assert "DISPLAY" in str(failure.value)


def test_a_detached_device_is_released_rather_than_left_open(ctx, monkeypatch):
    """``on_exit`` never runs for a node that failed to enter."""
    device = DetachedDevice()
    task = _drive_real_on_enter(monkeypatch, device)

    with pytest.raises(RuntimeError):
        task.on_enter(ctx, Drive.Inputs())

    assert device.closed


def test_an_attached_device_still_enters_normally(ctx, monkeypatch):
    device = ScriptedDevice([np.full(6, 0.2, dtype=np.float32)])
    task = _drive_real_on_enter(monkeypatch, device)

    task.on_enter(ctx, Drive.Inputs())

    assert device.opened
    assert task.tick(ctx) is Status.RUNNING


def test_a_device_that_says_nothing_is_assumed_attached():
    """Every device that raises on failure keeps working unchanged.

    Only the two keyboards downgrade a failure to a warning, so the default has
    to be true or the bus, leader and VR devices would all have to opt in to
    being usable.
    """
    assert ScriptedDevice().attached is True
    assert BusDevice("k").attached is True


def test_the_catheter_keyboard_is_detached_until_it_subscribes():
    """No Kit in a CPU test, so ``open`` is the failing path by construction."""
    assert CatheterKeyboardDevice().attached is False


def test_the_isaac_keyboard_is_detached_without_kit():
    assert KeyboardDevice().attached is False


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

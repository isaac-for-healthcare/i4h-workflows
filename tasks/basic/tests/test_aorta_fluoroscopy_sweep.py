# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import numpy as np
import pytest

from i4h_engine.status import Status
from i4h_tasks.basic.medical.aorta_fluoroscopy_sweep import AortaFluoroscopySweep


def context(envs=1):
    commands = []
    act = SimpleNamespace(
        dof=3,
        action_space="catheter_normalized_velocity",
        set_raw_action=lambda a, robot="robot": commands.append(a.copy()),
    )
    return SimpleNamespace(act=act, num_envs=envs, dt=1 / 30), commands


@pytest.mark.parametrize("envs", [1, 2])
def test_all_commands_phase_boundaries_and_terminal_behavior(envs):
    ctx, commands = context(envs)
    task = AortaFluoroscopySweep()
    task.on_enter(ctx, None)
    for index in range(115):
        assert task.tick(ctx) is (Status.SUCCESS if index == 114 else Status.RUNNING)
    expected = np.zeros((115, 3), dtype=np.float32)
    expected[15:45, 0] = 1
    expected[45:60, 1] = 0.5
    expected[60:75, 2] = 0.5
    expected[75:90, 2] = -0.5
    expected[90:100, 0] = -1
    np.testing.assert_array_equal(np.asarray(commands), np.repeat(expected[:, None, :], envs, axis=1))
    assert task.advance_on_success
    assert task.tick(ctx) is Status.SUCCESS
    assert len(commands) == 115
    assert task.on_exit(ctx).commands == 115


def test_task_restart_and_new_instance_have_fresh_state():
    ctx, commands = context()
    task = AortaFluoroscopySweep()
    task.on_enter(ctx, None)
    for _ in range(40):
        task.tick(ctx)
    for candidate in (task, AortaFluoroscopySweep()):
        candidate.on_enter(ctx, None)
        for _ in range(15):
            assert candidate.tick(ctx) is Status.RUNNING
            np.testing.assert_array_equal(commands[-1], 0)
        candidate.tick(ctx)
        np.testing.assert_array_equal(commands[-1], [[1, 0, 0]])


def test_engine_requests_physics_for_the_final_command():
    from i4h_engine.executor import Engine
    from i4h_engine.graph import TaskGraph, node
    from i4h_engine.status import WorkflowStatus
    from i4h_engine.task import TickContext
    from i4h_tasks.basic.testing.fake_scene import fake_world

    scene, act = fake_world(dof=3)
    ctx = TickContext(scene=scene, act=act, dt=1 / 30)
    engine = Engine(TaskGraph().flow(node(AortaFluoroscopySweep())))
    engine.start(ctx)
    ticks = 0
    while not engine.status.is_terminal:
        engine.tick(ctx)
        ticks += 1
    assert ticks == 115
    assert engine.status is WorkflowStatus.SUCCEEDED
    assert engine.terminal_advance_requested

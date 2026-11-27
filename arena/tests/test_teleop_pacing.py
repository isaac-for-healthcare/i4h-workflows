# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Deadline pacing without Kit, wall-clock waits or hardware."""

from types import SimpleNamespace

import pytest
from test_native_teleop_performance import ROOT, functions


class Clock:
    def __init__(self):
        self.now = 0.0
        self.waits = []

    def monotonic(self):
        return self.now

    def sleep(self, duration):
        assert duration > 0
        self.waits.append(duration)
        self.now += duration


def limiter(hz=30):
    clock = Clock()
    ns = functions(
        ROOT / "third_party/IsaacLab-3939445/scripts/tools/record_demos.py", {"RateLimiter"}, {"time": clock}
    )
    sim = SimpleNamespace(render_generation=0, renders=0)

    def render():
        sim.renders += 1
        sim.render_generation += 1
        clock.now += 0.002

    sim.render = render
    return ns["RateLimiter"](hz), clock, SimpleNamespace(sim=sim)


def test_sustained_fast_work_respects_rate_and_renders_once():
    pace, clock, env = limiter()
    for _ in range(300):
        pace.begin(env)
        clock.now += 0.005
        env.sim.render()
        pace.sleep(env)
    assert clock.now == pytest.approx(10.0)
    assert env.sim.renders == 300
    assert len(clock.waits) == 300


@pytest.mark.parametrize("step_duration", [0.04, 0.1, 0.105, 1.0])
def test_overrun_never_sleeps_or_adds_an_extra_period(step_duration):
    pace, clock, env = limiter(10)
    pace.begin(env)
    clock.now += step_duration
    env.sim.render_generation += 1
    pace.sleep(env)
    if step_duration >= 0.1:
        assert not clock.waits
        assert step_duration < pace.next_deadline <= step_duration + 0.1 + 1e-12
    else:
        assert clock.now == pytest.approx(0.1)
    assert env.sim.renders == 0
    pace.begin(env)
    clock.now += 0.001
    pace.sleep(env)
    assert env.sim.renders == 1
    assert clock.now == pytest.approx(pace.next_deadline - 0.1)


def test_paused_missing_action_and_resume_use_same_deadlines():
    pace, clock, env = limiter(20)
    for step in [False, False, True, False, True]:
        pace.begin(env)
        if step:
            clock.now += 0.01
            env.sim.render()
        pace.sleep(env)
    assert clock.now == pytest.approx(0.25)
    assert env.sim.renders == 5


def test_idle_sleep_without_begin_pumps_once_in_completion_delay():
    pace, clock, env = limiter()
    pace.sleep(env)
    pace.sleep(env)
    assert env.sim.renders == 2
    assert clock.now == pytest.approx(2 / 30)

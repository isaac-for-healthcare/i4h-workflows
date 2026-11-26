# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from types import SimpleNamespace

import pytest
import torch
import warp as wp

from i4h_arena.medical.liver import LiverObjective, encode_observation, normalized_action


class Fixture:
    def __init__(self, worlds=2, objective="full_cycle"):
        wp.init()
        wp.set_device("cpu")
        E = worlds
        self.task = LiverObjective(E, "cpu", objective)
        self.task.workspace[:] = torch.tensor([[[-1.0, -1.0, -1.0], [1.0, 1.0, 1.0]]])
        self.p = SimpleNamespace(
            binding=SimpleNamespace(scene=SimpleNamespace(device="cpu")),
            geometry=wp.zeros((E, 4), dtype=wp.vec3),
            counts=wp.zeros((E, 6), dtype=int),
            latched=wp.zeros(E, dtype=int),
            mechanism_counts=wp.zeros((E, 4), dtype=int),
            gap=wp.full(E, 0.01),
            ic=SimpleNamespace(actions=wp.zeros((E, 1, 10)), root=wp.zeros((E, 1), dtype=wp.vec3)),
        )
        self.q = wp.zeros((E, 3), dtype=wp.vec3)

    def sample(self, n=1):
        for _ in range(n):
            self.task.sample(self.p, self.q)

    def acquire(self):
        wp.to_torch(self.p.ic.actions)[:, 0, 7] = 0.6
        wp.to_torch(self.p.latched)[:] = 1
        wp.to_torch(self.p.counts)[:, :4] = torch.tensor([1, 1, 1, 1])
        self.sample()


def test_cap_diagonal_and_hold_mapping():
    x = normalized_action(torch.tensor([[1.0, 1.0, 1.0, 5.0], [0.0, 0.0, 0.0, -2.0]]))
    assert torch.linalg.vector_norm(x[0, :3]) <= 1.0
    assert x[0, 3] == 1 and x[1, 3] == -1


def test_full_cycle_and_both_substeps():
    f = Fixture()
    f.acquire()
    f.sample(30)
    assert torch.all(f.task.s[:, 0] == 3)
    wp.to_torch(f.p.geometry)[:, 0, 1] = 0.02
    f.sample(59)
    assert torch.all(f.task.s[:, 0] == 4)
    f.sample()
    assert torch.all(f.task.s[:, 0] == 5)
    wp.to_torch(f.p.ic.actions)[:, 0, 7] = 0
    f.sample()
    assert torch.all(f.task.s[:, 0] == 6)
    f.sample(60)
    assert torch.all(f.task.s[:, 0] == 7)


@pytest.mark.parametrize(
    "scenario,reason", [("wrong", 4), ("lost", 5), ("open", 9), ("deadline", 3), ("invalid", 1), ("workspace", 2)]
)
def test_failures_never_success(scenario, reason):
    f = Fixture()
    f.acquire()
    if scenario == "wrong":
        wp.to_torch(f.p.counts)[:, 5] = 1
    if scenario == "lost":
        wp.to_torch(f.p.counts)[:, 3] = 0
    if scenario == "open":
        wp.to_torch(f.p.ic.actions)[:, 0, 7] = 0
    if scenario == "deadline":
        f.task.s[:, 1] = 1199
    if scenario == "invalid":
        wp.to_torch(f.q)[:, 0, 0] = float("nan")
    if scenario == "workspace":
        wp.to_torch(f.p.ic.root)[:, 0, 0] = 2
    f.sample()
    assert torch.all(f.task.s[:, 0] == 8)
    assert torch.all(f.task.s[:, 8] == reason)


def test_interrupted_hold_release_timeout_and_relaxation():
    f = Fixture()
    f.acquire()
    f.sample(30)
    wp.to_torch(f.p.geometry)[:, 0, 1] = 0.02
    f.sample(59)
    wp.to_torch(f.p.geometry)[:, 0, 1] = 0.023
    f.sample()
    assert torch.all(f.task.s[:, 3] == 0)
    wp.to_torch(f.p.geometry)[:, 0, 1] = 0.02
    f.sample(60)
    wp.to_torch(f.p.ic.actions)[:, 0, 7] = 0
    wp.to_torch(f.p.mechanism_counts)[:, 3] = 1
    f.sample(31)
    assert torch.all(f.task.s[:, 8] == 7)


def test_curriculum_is_identified_and_reset_masked():
    f = Fixture(objective="reach_only")
    wp.to_torch(f.p.gap)[:] = -0.00025
    f.sample(2)
    assert torch.all(f.task.s[:, 0] == 7)
    peer = f.task.s[1].clone()
    f.task.reset(torch.tensor([0]))
    assert torch.equal(f.task.s[1], peer)
    rows = f.task.results([1], [252, 257], wp.to_torch(f.p.geometry))
    assert rows[0]["objective"] == "reach_only" and rows[0]["success"]


def test_encoder_has_reference_values_and_is_pure():
    E = 2
    geometry = torch.arange(E * 12, dtype=torch.float32).reshape(E, 4, 3) * 0.001
    root = torch.full((E, 3), 0.002)
    action = torch.zeros(E, 10)
    action[:, 6] = 1
    s = torch.zeros(E, 9, dtype=torch.int32)
    s[:, 4] = -1
    f = torch.zeros(E, 10)
    args = (
        geometry,
        root,
        action,
        torch.zeros(E, 2),
        torch.zeros(E, 6, dtype=torch.int32),
        torch.zeros(E, 2),
        torch.zeros(E, 4, dtype=torch.int32),
        torch.zeros(E, 4),
        geometry[:, 0].clone(),
        root.clone(),
        s,
        f,
    )
    before = s.clone()
    obs = encode_observation(*args)
    assert obs.shape == (2, 57)
    torch.testing.assert_close(obs[:, :3], (geometry[:, 0] - root) * 100)
    torch.testing.assert_close(obs[:, 12:18], torch.zeros(E, 6))
    torch.testing.assert_close(obs[:, 18:21], torch.tensor([[0.0, 2.0, 0.0]]).repeat(E, 1))
    assert obs[0, 44] == 1  # approach one-hot
    assert torch.equal(obs, encode_observation(*args)) and torch.equal(s, before)


def test_reacquisition_during_relaxation_and_acquisition_on_first_substep():
    f = Fixture()
    f.acquire()
    assert torch.all(f.task.s[:, 4] == 1)
    f.sample(30)
    wp.to_torch(f.p.geometry)[:, 0, 1] = 0.02
    f.sample(60)
    wp.to_torch(f.p.ic.actions)[:, 0, 7] = 0
    f.sample()
    wp.to_torch(f.p.mechanism_counts)[:, 2] = 1
    f.sample()
    assert torch.all(f.task.s[:, 8] == 8)


def test_empty_capture_and_terminal_metrics_preserve_final_and_maximum():
    f = Fixture()
    wp.to_torch(f.p.ic.actions)[:, 0, 7] = 0.6
    f.sample(61)
    assert torch.all(f.task.s[:, 8] == 6)
    f = Fixture()
    f.acquire()
    f.task.f[:, 6] = 0.024
    wp.to_torch(f.p.geometry)[:, 0, 1] = 0.019
    rows = f.task.results([0], [252, 257], wp.to_torch(f.p.geometry))
    assert rows[0]["max_lift_m"] > rows[0]["actual_lift_m"]

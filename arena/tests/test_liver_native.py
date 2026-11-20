# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Native policy commands and complete-state recorder serialization."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from isaaclab_arena.policy.policy_base import PolicyCfg

from i4h_arena.medical.liver_recording import record_bundle, replay_bundle
from i4h_arena.policies.aorta import AortaFluoroscopySweepPolicy
from i4h_arena.policies.hold import _reset_state_action
from i4h_arena.policies.liver import LiverReferencePolicy


def host(**kwargs):
    result = SimpleNamespace(num_envs=2, device="cpu", **kwargs)
    result.unwrapped = result
    return result


def test_idle_holds_position_and_opens_jaws():
    env = host(
        cfg=SimpleNamespace(idle_action=[0.0, 0.0, 0.0, -1.0]), action_manager=SimpleNamespace(total_action_dim=4)
    )
    action = _reset_state_action(env)
    torch.testing.assert_close(action, torch.tensor([[0.0, 0.0, 0.0, -1.0]] * 2))


def test_reference_reads_host_and_rejects_curriculum():
    liver = SimpleNamespace(
        objective=SimpleNamespace(objective="full_cycle"), teacher_action=lambda: np.ones((2, 4), np.float32)
    )
    policy = LiverReferencePolicy(PolicyCfg())
    torch.testing.assert_close(policy.get_action(host(liver=liver), None), torch.ones(2, 4))
    liver.objective.objective = "reach_only"
    with pytest.raises(ValueError, match="full-cycle"):
        policy.get_action(host(liver=liver), None)


def test_static_aorta_commands_and_selected_policy_reset():
    policy = AortaFluoroscopySweepPolicy(PolicyCfg())
    env = host()
    expected = [command for count, command in policy.phases for _ in range(count)]
    assert len(expected) == 115
    for command in expected:
        torch.testing.assert_close(policy.get_action(env, None), torch.tensor([command] * 2, dtype=torch.float32))
    policy.reset(torch.tensor([1]))
    for _ in range(16):
        action = policy.get_action(env, None)
    torch.testing.assert_close(action, torch.tensor([[0.0, 0.0, 0.0], [1.0, 0.0, 0.0]]))


def test_initial_state_round_trip_keeps_dtype_and_full_model_layout():
    source = {
        "identity": np.arange(64, dtype=np.uint8)[None],
        "numerical": {"state:q": np.arange(18, dtype=np.float32).reshape(6, 3)},
        "task": {"phase": np.array([[2], [3]], dtype=np.int32)},
    }
    packed = record_bundle(source, 2)

    def episode(value):
        return {key: episode(item) for key, item in value.items()} if isinstance(value, dict) else value[1].unsqueeze(0)

    restored = replay_bundle(episode(packed))
    np.testing.assert_array_equal(restored["identity"], source["identity"])
    for family in ("numerical", "task"):
        for key in source[family]:
            np.testing.assert_array_equal(restored[family][key], source[family][key])
            assert restored[family][key].dtype == source[family][key].dtype

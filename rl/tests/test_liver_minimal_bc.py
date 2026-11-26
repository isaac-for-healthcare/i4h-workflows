# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""CPU checks for the minimal liver RL example's demonstration and BC stages."""

from types import SimpleNamespace

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from i4h_rl.liver_experiment import (
    collect_demonstrations,
    evaluate,
    liver_actor,
    prepare_starts,
    select_checkpoint,
    train_bc,
)


def _demonstrations(path, episodes=12, steps=20, seed=3):
    rng = np.random.default_rng(seed)
    weights = rng.normal(size=(57, 4)).astype(np.float32) * 0.1
    observations = rng.normal(size=(episodes * steps, 57)).astype(np.float32)
    actions = np.tanh(observations @ weights)
    episode = np.repeat(np.arange(episodes), steps)
    np.savez_compressed(path, observations=observations, actions=actions, episode=episode)
    return observations, actions


def test_train_bc_writes_an_import_compatible_actor_that_fits_its_demonstrations(tmp_path):
    observations, actions = _demonstrations(tmp_path / "demos.npz")
    report = train_bc(tmp_path / "demos.npz", tmp_path / "bc", epochs=80, batch_size=64, seed=1)
    assert report["episodes"] == 12 and report["validation_episodes"] == 1
    first, last = report["history"][0], report["history"][-1]
    assert last["train_mse"] < first["train_mse"] * 0.5
    assert last["validation_mse"] is not None
    state = torch.load(tmp_path / "bc" / "bc.pt", weights_only=False)
    frozen = np.load(tmp_path / "bc" / "observations.npz")["observations"]
    assert frozen.shape[1] == 57 and len(frozen) <= len(observations)
    td = TensorDict({"policy": torch.as_tensor(frozen)}, batch_size=[len(frozen)])
    # import_bc constructs the same actor and loads it strictly.
    actor = liver_actor(td)
    actor.load_state_dict(state["actor"], strict=True)
    with torch.no_grad():
        mean = actor(TensorDict({"policy": torch.as_tensor(observations)}, batch_size=[len(observations)]))
    assert float(torch.nn.functional.mse_loss(mean, torch.as_tensor(actions))) < first["train_mse"]


def test_train_bc_rejects_malformed_demonstrations(tmp_path):
    np.savez_compressed(tmp_path / "bad.npz", observations=np.zeros((4, 10)), actions=np.zeros((4, 4)), episode=[0] * 4)
    with pytest.raises(ValueError, match="57"):
        train_bc(tmp_path / "bad.npz", tmp_path / "bc", epochs=1)


class _FakeLiver:
    objective = SimpleNamespace(objective="full_cycle")

    def __init__(self, env):
        self.env = env
        self.next_specs = {}
        self.q = SimpleNamespace(numpy=lambda: np.zeros((env.num_envs, 3)))
        self.physical = torch.zeros(env.num_envs, 8)

    def observation(self):
        return torch.full((self.env.num_envs, 57), float(self.env.tick))

    def teacher_action(self):
        return torch.full((self.env.num_envs, 4), 0.5)


class _FakeEnv:
    """Each world's episode lasts ``2 + world`` ticks; world 1 fails."""

    device = "cpu"

    def __init__(self, num_envs=2):
        self.num_envs = num_envs
        self.tick = 0
        self.length = {}
        self.executed = []
        self.liver = _FakeLiver(self)

    def reset(self, env_ids):
        for e in env_ids.tolist():
            self.length[e] = 0

    def step(self, action):
        self.executed.append(action.clone())
        self.tick += 1
        rows = []
        for e in list(self.length):
            self.length[e] += 1
            if self.length[e] == 2 + e:
                rows.append(dict(env_id=e, objective="full_cycle", target=0, success=e == 0, failure_reason=""))
        return None, None, None, None, {"terminal_results": rows}


def test_demonstrations_cover_every_world_and_keep_clean_labels():
    env = _FakeEnv()
    specs = [dict(target_index=0, jitter_m=[0.0, 0.0, 0.0])] * 4
    episodes = []
    report = evaluate(env, specs, demonstrations=episodes, action_noise=0.2, noise_seed=5)
    assert report["episodes"] == 4 and report["successes"] == 2
    assert sorted(e["evaluation_id"] for e in episodes) == [0, 1, 2, 3]
    for episode in episodes:
        world = episode["evaluation_id"] % env.num_envs
        assert episode["observations"].shape == (2 + world, 57)
        assert episode["success"] is (world == 0)
        # Recorded labels are the clean reference action, not the executed noisy one.
        np.testing.assert_array_equal(episode["actions"], np.full((2 + world, 4), 0.5, dtype=np.float32))
    executed = torch.stack(env.executed)
    assert not torch.allclose(executed[..., :3], torch.full_like(executed[..., :3], 0.5))
    torch.testing.assert_close(executed[..., 3], torch.full_like(executed[..., 3], 0.5))


def test_collect_demonstrations_creates_its_output_directory_and_keeps_successes(tmp_path):
    args = SimpleNamespace(
        episodes=4, seed=7, action_noise=0.0, successful_only=True, output=tmp_path / "new" / "demonstrations.json"
    )
    report = collect_demonstrations(_FakeEnv(), args)
    assert report["kept_episodes"] == 2 and report["successes"] == 2
    saved = np.load(tmp_path / "new" / "demonstrations.npz")
    # World 0 succeeds with 2-tick episodes; world 1's 3-tick failures are dropped.
    assert saved["observations"].shape == (4, 57) and saved["actions"].shape == (4, 4)
    assert len(np.unique(saved["episode"])) == 2


def test_evaluate_creates_the_trace_directory(tmp_path):
    specs = [dict(target_index=0, jitter_m=[0.0, 0.0, 0.0])] * 2
    evaluate(_FakeEnv(), specs, trace_path=tmp_path / "evaluation" / "bc.npz")
    assert (tmp_path / "evaluation" / "bc.npz").is_file()


def test_selection_uses_development_success_and_earliest_tie(tmp_path):
    import json

    prepare_starts(tmp_path / "starts", 40)
    dev = tmp_path / "starts/development.json"
    starts = json.loads(dev.read_text())
    tests = json.loads((tmp_path / "starts/test.json").read_text())
    assert len(starts) == 20 and len(tests) == 40
    assert not {s["seed"] for s in starts} & {s["seed"] for s in tests}
    candidates = []
    for update, successes in [(30, 20), (6, 14), (15, 20), (10, 19)]:
        checkpoint = tmp_path / f"update_{update}.pt"
        checkpoint.write_text(str(update))
        candidates.append(dict(update=update, successes=successes, episodes=20, checkpoint=str(checkpoint)))
    output = tmp_path / "selection/selection.json"
    select_checkpoint(candidates, output, dev)
    selected = json.loads(output.read_text())
    assert selected["selected_update"] == 15
    assert (tmp_path / "selection/model_selected.pt").read_text() == "15"
    with pytest.raises(ValueError, match="20 development"):
        select_checkpoint([dict(candidates[0], episodes=10)], output, dev)

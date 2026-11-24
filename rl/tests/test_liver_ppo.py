# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
import copy
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from i4h_rl.liver_ppo import GuardedPPO, MinimalLiverPPO


def algorithm(warmup=10, lr=1e-5, cls=GuardedPPO):
    torch.manual_seed(17)
    obs = TensorDict({"policy": torch.randn(8, 57)}, batch_size=[8])
    cfg = dict(
        actor=dict(
            class_name="rsl_rl.models:MLPModel",
            hidden_dims=[128, 128],
            activation="elu",
            obs_normalization=False,
            distribution_cfg=dict(class_name="rsl_rl.modules:GaussianDistribution", init_std=0.02),
        ),
        critic=dict(
            class_name="rsl_rl.models:MLPModel", hidden_dims=[128, 128], activation="elu", obs_normalization=False
        ),
        algorithm=dict(
            class_name=f"i4h_rl.liver_ppo:{cls.__name__}",
            **({} if cls is MinimalLiverPPO else dict(qualification_required=False)),
            critic_warmup_updates=warmup,
            learning_rate=lr,
            schedule="fixed",
            num_learning_epochs=2,
            num_mini_batches=2,
        ),
        obs_groups=dict(actor=["policy"], critic=["policy"]),
        num_steps_per_env=8,
        multi_gpu=None,
    )
    alg = cls.construct_algorithm(obs, SimpleNamespace(num_envs=8, num_actions=4), cfg, "cpu")
    return alg, obs


def rollout(alg, obs):
    with torch.no_grad():
        for _ in range(8):
            alg.act(obs)
            next_obs = TensorDict({"policy": torch.randn(8, 57)}, batch_size=[8])
            alg.process_env_step(next_obs, torch.randn(8), torch.zeros(8, dtype=torch.bool), {})
            obs = next_obs
        alg.compute_returns(obs)


def test_zero_lr_preserves_complete_actor_and_distribution():
    alg, obs = algorithm(warmup=0, lr=0.0)
    # The unit test evaluator is explicit and independent of the update invariance check.
    alg.evaluator = lambda actor: dict(episodes=20, starts_sha256="frozen", objective="full_cycle", successes=20)
    alg.qualification = dict(validation_starts_sha256="frozen", objective="full_cycle", bc_successes=20)
    before = copy.deepcopy(alg.actor.state_dict())
    rollout(alg, obs)
    report = alg.update()
    assert report["update_kl"] < 1e-10
    assert all(torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items())


def test_critic_only_invariance_and_resume():
    alg, obs = algorithm()
    actor = copy.deepcopy(alg.actor.state_dict())
    critic = copy.deepcopy(alg.critic.state_dict())
    rollout(alg, obs)
    report = alg.update()
    assert report["critic_only"] == 1 and report["update_kl"] < 1e-10
    assert all(torch.equal(v, alg.actor.state_dict()[k]) for k, v in actor.items())
    assert any(not torch.equal(v, alg.critic.state_dict()[k]) for k, v in critic.items())
    saved = copy.deepcopy(alg.save())
    resumed, _ = algorithm()
    resumed.load(saved, None, True)
    assert resumed.completed_updates == 1 and resumed.actor_updates == 0
    assert all(torch.equal(v, resumed.actor.state_dict()[k]) for k, v in actor.items())


def test_large_kl_rolls_back_actor_and_optimizer_moments():
    alg, obs = algorithm(warmup=0, lr=0.1)
    alg.evaluator = lambda actor: dict(episodes=20, starts_sha256="frozen", objective="full_cycle", successes=20)
    alg.qualification = dict(validation_starts_sha256="frozen", objective="full_cycle", bc_successes=20)
    before = copy.deepcopy(alg.actor.state_dict())
    rollout(alg, obs)
    report = alg.update()
    assert report["update_kl"] > 0.02 and report["actor_rollback"] == 1
    assert all(torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items())
    assert all(p not in alg.optimizer.state for p in alg.actor.parameters())


def test_rollback_restores_existing_adam_moments():
    alg, obs = algorithm(warmup=0, lr=1e-8)
    alg.evaluator = lambda actor: dict(episodes=20, starts_sha256="frozen", objective="full_cycle", successes=20)
    alg.qualification = dict(validation_starts_sha256="frozen", objective="full_cycle", bc_successes=20)
    rollout(alg, obs)
    first = alg.update()
    assert first["actor_rollback"] == 0
    before = copy.deepcopy(alg.actor.state_dict())
    parameters = [p for p in alg.actor.parameters() if p in alg.optimizer.state]
    moments = {p: copy.deepcopy(alg.optimizer.state[p]) for p in parameters}
    assert moments
    alg.learning_rate = 0.1
    for group in alg.optimizer.param_groups:
        group["lr"] = 0.1
    rollout(alg, obs)
    result = alg.update()
    assert result["actor_rollback"] == 1
    assert all(torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items())
    for parameter, state in moments.items():
        for key, value in state.items():
            if isinstance(value, torch.Tensor):
                assert torch.equal(value, alg.optimizer.state[parameter][key])
            else:
                assert value == alg.optimizer.state[parameter][key]


def test_validation_stop_blocks_later_updates_and_pending_resume(tmp_path):
    import pytest

    alg, obs = algorithm(warmup=0, lr=0.0)
    alg.evaluator = lambda actor: dict(episodes=20, starts_sha256="frozen", objective="full_cycle", successes=17)
    alg.qualification = dict(validation_starts_sha256="frozen", objective="full_cycle", bc_successes=20)
    alg._validation_folder = tmp_path
    rollout(alg, obs)
    with pytest.raises(RuntimeError, match="stop training"):
        alg.update()
    stopped = torch.load(tmp_path / "stopped_checkpoint.pt", weights_only=False)
    assert stopped["liver_safeguards"]["stopped"] is True
    with pytest.raises(RuntimeError, match="stopped"):
        alg.update()
    fresh, _ = algorithm()
    with pytest.raises(RuntimeError, match="awaited mandatory evaluation"):
        fresh.load({"infos": {"validation_pending": True}}, None, True)


def test_guarded_ppo_still_requires_its_evaluator_for_actor_updates():
    import pytest

    alg, obs = algorithm(warmup=0)
    rollout(alg, obs)
    with pytest.raises(RuntimeError, match="BC comparison evaluator"):
        alg.update()


def test_minimal_ppo_updates_the_actor_without_qualification_or_evaluator():
    alg, obs = algorithm(warmup=1, lr=1e-6, cls=MinimalLiverPPO)
    assert isinstance(alg, MinimalLiverPPO) and alg.qualification is None and alg.evaluator is None
    rollout(alg, obs)
    assert alg.update()["critic_only"] == 1
    before = copy.deepcopy(alg.actor.state_dict())
    kept = 0
    for _ in range(5):
        rollout(alg, obs)
        report = alg.update()
        assert report["critic_only"] == 0 and "deterministic_successes" not in report
        kept += report["actor_rollback"] == 0
    assert kept
    assert alg.actor_updates == 5 and not alg.stopped
    assert any(not torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items())
    # The distribution stays frozen exactly as in the qualified campaign.
    for parameter in alg.actor.distribution.parameters():
        assert not parameter.requires_grad
    assert alg.save()["liver_safeguards"]["validation"] == "none (minimal example)"


def test_minimal_ppo_keeps_kl_rollback():
    alg, obs = algorithm(warmup=0, lr=0.1, cls=MinimalLiverPPO)
    before = copy.deepcopy(alg.actor.state_dict())
    rollout(alg, obs)
    report = alg.update()
    assert report["update_kl"] > 0.02 and report["actor_rollback"] == 1
    assert all(torch.equal(v, alg.actor.state_dict()[k]) for k, v in before.items())


def test_minimal_ppo_rejects_qualification():
    import pytest

    with pytest.raises(ValueError, match="unqualified"):
        MinimalLiverPPO(None, None, None, qualification_required=True)


def test_reference_gradient_matches_explicit_bc_kl_loss():
    alg, obs = algorithm(warmup=0, cls=MinimalLiverPPO)
    reference = alg._reference()
    with torch.no_grad():
        next(alg.actor.parameters()).add_(0.001)
    expected_actor = copy.deepcopy(alg.actor)
    expected_actor(obs, stochastic_output=True)
    mean, std = expected_actor.output_distribution_params
    with torch.no_grad():
        target = reference(obs)
    loss = mean.square().mean() + alg.bc_reference_coef * ((mean - target).square() / (2 * std.square())).sum(-1).mean()
    loss.backward()
    handle = alg.actor.register_forward_hook(alg._reference_gradient, with_kwargs=True)
    try:
        alg.actor(obs, stochastic_output=True)
        alg.actor.output_distribution_params[0].square().mean().backward()
    finally:
        handle.remove()
    for actual, expected in zip(alg.actor.parameters(), expected_actor.parameters(), strict=True):
        if actual.requires_grad:
            torch.testing.assert_close(actual.grad, expected.grad)
    assert all(p.grad is None and not p.requires_grad for p in reference.parameters())


def test_bc_reference_survives_actor_changes_and_resume():
    alg, obs = algorithm(warmup=0, lr=1e-6, cls=MinimalLiverPPO)
    before = copy.deepcopy(alg._reference().state_dict())
    rollout(alg, obs)
    alg.update()
    saved = copy.deepcopy(alg.save())
    assert any(not torch.equal(v, saved["actor_state_dict"][k]) for k, v in before.items())
    resumed, _ = algorithm(warmup=0, cls=MinimalLiverPPO)
    resumed.load(saved, None, True)
    assert resumed.completed_updates == 1
    assert all(torch.equal(v, resumed._reference().state_dict()[k]) for k, v in before.items())
    assert not alg.actor._forward_hooks


def test_resume_rejects_missing_or_changed_reference():
    import pytest

    alg, _ = algorithm(cls=MinimalLiverPPO)
    saved = copy.deepcopy(alg.save())
    saved["liver_bc_reference"]["coefficient"] = 0.2
    with pytest.raises(ValueError, match="coefficient mismatch"):
        alg.load(saved, None, True)
    del saved["liver_bc_reference"]
    saved["liver_safeguards"]["completed_updates"] = 1
    with pytest.raises(ValueError, match="original frozen BC reference"):
        alg.load(saved, None, True)

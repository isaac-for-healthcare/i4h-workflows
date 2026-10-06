# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Real CUDA kitless checks; use the isolated feasibility runtime, not stub modules."""

import numpy as np
import pytest
import torch
import warp as wp

pytest.importorskip("isaaclab_arena")
pytest.importorskip("i4h_endoluminal")

from i4h_arena.i4h_aorta_reach import AortaReachTask, close_env, make_env, reset_episode  # noqa: E402


@pytest.fixture(params=[(n, c, d) for n in (1, 2) for c in (False, True) for d in (1, 2, 4, 8)])
def env(request):
    count, captured, decimation = request.param
    instance = make_env(count, captured=captured, decimation=decimation, validate_evidence=True)
    try:
        instance.reset()
        yield instance
    finally:
        owner, evidence = instance.scene["catheter"].owner, instance.scene["catheter"].evidence
        close_env(instance)
        assert not owner.valid and not evidence.valid


def action(env, value=0.0):
    return torch.full((env.num_envs, 3), value, device=env.device)


@pytest.mark.parametrize("decimation", [0, -1, 1.5, True])
def test_invalid_decimation_rejected(decimation):
    from i4h_arena.i4h_aorta_reach import make_cfg

    with pytest.raises(ValueError, match="positive integer"):
        make_cfg(decimation=decimation)


def snapshot(asset):
    asset.to_torch()
    return [
        a.clone()
        for a in (
            asset.q,
            asset.qd,
            asset.flags,
            asset.commands,
            asset.targets,
            asset.rows,
            asset.history,
            asset.progress,
        )
    ]


def physical_history(asset):
    asset.to_torch()
    scene = asset.sim.scene
    body = scene.body("catheter")
    rod = next(s for s in asset.sim.systems if hasattr(s, "young"))
    contact = next(s for s in asset.sim.systems if hasattr(s, "contact_count"))
    arrays = [wp.to_torch(scene.field("q_prev", body))]
    for array in (rod.frame, rod.frame_pred, rod.omega, rod.bend_offset, contact.contact_depth, contact.contact_count):
        arrays.append(wp.to_torch(array).reshape(scene.envs, -1))
    return [array.clone() for array in arrays]


def test_construction_actions_observations_and_read_purity(env):
    asset = env.scene["catheter"]
    assert asset.spec == AortaReachTask.spec
    assert asset.sim.procedure is None and "procedure" not in env.cfg.sim.physics.solver_cfg.case
    assert asset.evidence.capacity == env.cfg.decimation and not env.recorder_manager.active_terms
    env.reset()  # Repeated public manual reset with recording disabled.
    assert asset.q.shape == (env.num_envs, 192, 3)
    assert asset.flags.shape == (env.num_envs, 192)
    before = snapshot(asset)
    elapsed = asset.rows[:, 9].clone()
    term = env.action_manager.get_term("catheter")
    term.process_actions(action(env, 2.0))
    term.apply_actions()
    asset.to_torch()
    assert torch.equal(asset.commands, action(env, 1.0))
    assert torch.equal(asset.rows[:, 9], elapsed), "Action terms never step physics or assessment"
    reset_episode(env, torch.arange(env.num_envs, device=env.device))
    before = snapshot(asset)
    for _ in range(3):
        obs = env.observation_manager.compute()
    after = snapshot(asset)
    assert all(torch.equal(a, b) for a, b in zip(before, after))
    assert obs["policy"].shape == (env.num_envs, 1347)
    assert torch.equal(obs["task"], asset.rows)
    command = action(env)
    command[:] = torch.tensor([0.2, 0.3, -0.1], device=env.device)
    obs, reward, term, trunc, _ = env.step(command)
    assert torch.equal(asset.commands, command)
    assert torch.allclose(
        asset.targets[0], torch.tensor([0.2 * 0.5, 0.3 * 1.5, -0.1 * 1.0], device=env.device) * env.step_dt
    )
    assert torch.isfinite(obs["policy"]).all()
    assert not term.any() and not trunc.any() and not reward.any()
    assert torch.allclose(asset.rows[:, 9], torch.full((env.num_envs,), env.step_dt, device=env.device))
    assert np.all(asset.evidence.count.numpy() == env.cfg.decimation)
    assert not asset.evidence.overflow.numpy().any()
    assert np.array_equal(asset.evidence.positions.numpy()[:, -1], asset.physics.q.numpy()[:, -1])
    before = snapshot(asset)
    asset.update(env.step_dt)  # duplicate lifecycle call consumes no samples twice
    assert all(torch.equal(a, b) for a, b in zip(before, snapshot(asset)))


def test_complete_and_partial_reset_and_rebuild(env):
    asset = env.scene["catheter"]
    initial = snapshot(asset)
    initial_history = physical_history(asset)
    rod = next(s for s in asset.sim.systems if hasattr(s, "young"))
    wp.to_torch(rod.young).mul_(1.01)
    young = wp.to_torch(rod.young).clone()
    asset.to_warp()
    for _ in range(3):
        env.step(action(env, 0.2))
    stepped = snapshot(asset)
    stepped_history = physical_history(asset)
    ids = torch.tensor([0], device=env.device)
    reset_episode(env, ids)
    reset = snapshot(asset)
    reset_history = physical_history(asset)
    for old, new in zip(initial[:5], reset[:5]):
        assert torch.equal(old[0], new[0])
    assert asset.rows[0, 9] == 0 and asset.rows[0, 11] == 0
    assert torch.equal(wp.to_torch(rod.young), young)
    assert all(torch.equal(a[0], b[0]) for a, b in zip(initial_history, reset_history))
    if env.num_envs == 2:
        assert all(torch.equal(a[1], b[1]) for a, b in zip(stepped, reset))
        assert all(torch.equal(a[1], b[1]) for a, b in zip(stepped_history, reset_history))
    old_binding = asset.physics
    old_owner, old_evidence = asset.owner, asset.evidence
    old_ptr, old_assessment = asset.q.data_ptr(), asset.assessment
    env.sim.reset(soft=False)
    assert not old_binding.valid
    with pytest.raises(RuntimeError, match="invalidated"):
        old_owner.reset()
    assert not old_evidence.valid
    assert asset.q.data_ptr() != old_ptr and asset.assessment is not old_assessment
    assert torch.equal(asset.q, initial[0])
    assert not env.action_manager.get_term("catheter").raw_actions.any()
    obs, *_ = env.step(action(env, 0.2))
    assert torch.isfinite(obs["policy"]).all()


def set_test_target_at_tip(asset):
    """Inject an assessment target for lifecycle checks; this does not qualify navigation."""
    asset.to_torch()
    asset.assessment.steps[0].fixed = wp.vec3(*asset.q[0, -1].cpu().tolist())


def test_hold_interruption_and_terminal_export(env, monkeypatch):
    decimation = env.cfg.decimation
    asset = env.scene["catheter"]
    # The initial rod settles into the vessel; use the settled tip for the synthetic target.
    for _ in range(64 // decimation):
        env.step(action(env))
    set_test_target_at_tip(asset)
    for _ in range(8 // decimation):
        _, reward, term, trunc, _ = env.step(action(env))
        assert not reward.any() and not term.any() and not trunc.any()
    assert torch.allclose(asset.rows[:, 10], torch.full_like(asset.rows[:, 10], 8 * env.physics_dt))
    asset.assessment.steps[0].fixed = wp.vec3(100, 100, 100)
    env.step(action(env))
    assert not asset.rows[:, 10].any()
    set_test_target_at_tip(asset)
    boundary = []
    update = asset.update

    def save_boundary(dt):
        update(dt)
        asset.to_torch()
        boundary.append(asset.q.clone())

    monkeypatch.setattr(asset, "update", save_boundary)
    for _ in range((30 + decimation - 1) // decimation - 1):
        _, reward, term, trunc, _ = env.step(action(env))
        assert not term.any() and not trunc.any() and not reward.any()
    _, reward, term, trunc, extras = env.step(action(env))
    assert term.all() and not trunc.any()
    assert torch.allclose(reward, torch.ones_like(reward))
    assert extras["new_terminal"].all() and (extras["terminal_task"][:, 11] == 1).all()
    completion_time = (64 + 8 + decimation + 30) * env.physics_dt
    assert torch.allclose(extras["terminal_task"][:, 9], torch.full_like(asset.rows[:, 9], completion_time))
    assert torch.allclose(extras["terminal_history"][:, 0, 1], torch.full_like(asset.history[:, 0, 1], completion_time))
    assert torch.equal(extras["final_obs"]["policy"][:, :576], boundary[-1].flatten(1))
    assert (extras["final_obs"]["task"][:, 11] == 1).all()
    terminal = {key: extras[key] for key in ("terminal_task", "terminal_history", "terminal_progress", "new_terminal")}
    assert not asset.rows[:, 9].any()  # Assessment resets with physics during autoreset.
    _, reward, term, trunc, _ = env.step(action(env))
    assert not reward.any() and not term.any() and not trunc.any()
    reset_episode(env, torch.arange(env.num_envs, device=env.device))
    for key in ("terminal_task", "terminal_history", "terminal_progress"):
        assert torch.equal(env.extras[key], terminal[key])
    assert terminal["new_terminal"].all()  # Earlier exports are copies across running resets.
    assert not env.extras["new_terminal"].any()


@pytest.mark.parametrize("complete", [True, False])
def test_deadline_precedence(env, complete):
    asset = env.scene["catheter"]
    if complete:
        set_test_target_at_tip(asset)
    wp.to_torch(asset.assessment.elapsed).fill_(120 - env.physics_dt)
    wp.to_torch(asset.assessment.active_hold).fill_(0.5 - env.physics_dt)
    asset.to_warp()
    _, reward, term, trunc, extras = env.step(action(env))
    assert bool(term.all()) == complete
    assert bool(trunc.all()) != complete
    assert torch.allclose(reward, torch.full_like(reward, float(complete)))
    assert (extras["terminal_task"][:, 11] == (1 if complete else 2)).all()
    assert np.allclose(extras["terminal_task"][:, 9].cpu().numpy(), 120)


def test_inactive_tip_and_side_stream_reset(env):
    asset = env.scene["catheter"]
    set_test_target_at_tip(asset)
    stream = torch.cuda.Stream(device=env.device)
    with torch.cuda.stream(stream):
        asset.flags[:, -1].bitwise_and_(~1)
        _, reward, term, trunc, _ = env.step(action(env))
        assert not reward.any() and not term.any() and not trunc.any()
        assert not asset.rows[:, 10].any()
        reset_episode(env, torch.arange(env.num_envs, device=env.device))
        asset.to_torch()
        assert (asset.flags[:, -1] & 1).all()
        assert not asset.targets.any() and not asset.commands.any()
    torch.cuda.current_stream(env.device).wait_stream(stream)


def test_timeout_notification_is_one_action_pulse(env):
    asset = env.scene["catheter"]
    wp.to_torch(asset.assessment.elapsed).fill_(120 - env.physics_dt)
    asset.to_warp()
    _, _, _, trunc, extras = env.step(action(env))
    assert trunc.all() and extras["new_terminal"].all()
    exported = {
        key: extras[key].clone() for key in ("terminal_task", "terminal_history", "terminal_progress", "new_terminal")
    }
    _, _, term, trunc, extras = env.step(action(env))
    assert not term.any() and not trunc.any() and not extras["new_terminal"].any()
    reset_episode(env, torch.arange(env.num_envs, device=env.device))
    assert not extras["new_terminal"].any()
    for key in ("terminal_task", "terminal_history", "terminal_progress"):
        assert torch.equal(extras[key], exported[key])
    assert exported["new_terminal"].all(), "Earlier exported payloads must remain copies"


def test_staggered_notifications_and_duplicate_export(env):
    if env.num_envs != 2:
        pytest.skip("Requires two environments")
    asset = env.scene["catheter"]
    for index in (0, 1):
        wp.to_torch(asset.assessment.elapsed)[index] = 120 - env.physics_dt
        asset.to_warp()
        _, _, _, trunc, extras = env.step(action(env))
        expected = [index == 0, index == 1]
        assert trunc.tolist() == expected
        assert extras["new_terminal"].tolist() == expected
    assert (extras["terminal_task"][:, 11] == 2).all()
    # Export twice before a physical reset: only the first export is fresh.
    ids = torch.tensor([0], device=env.device)
    asset.rows[0, 11] = 2
    asset.recorded[0] = True
    asset.new_terminal.zero_()
    reset_episode(env, ids)
    assert not env.extras["new_terminal"].any()


def test_failed_construction_cleans_hook_and_callback():
    from i4h_isaaclab.newton_manager import I4hNewtonManager, construction_context
    from isaaclab_newton.physics import NewtonManager

    from i4h_arena.i4h_aorta_reach import make_cfg

    cfg, _ = make_cfg()
    previous_hooks = list(NewtonManager._per_world_builder_hooks)
    previous_callbacks = list(I4hNewtonManager.on_solver)

    def callback(sim):
        pass

    with pytest.raises(RuntimeError, match="construction failed"):
        with construction_context(
            cfg.sim.physics.solver_cfg, device=cfg.sim.device, envs=cfg.scene.num_envs, on_solver=callback
        ):
            raise RuntimeError("construction failed")
    assert NewtonManager._per_world_builder_hooks == previous_hooks
    assert I4hNewtonManager.on_solver == previous_callbacks


def test_constructor_failure_releases_runtime(monkeypatch):
    from i4h_isaaclab.newton_manager import I4hNewtonManager
    from isaaclab.sim import SimulationContext
    from isaaclab_newton.physics import NewtonManager

    from i4h_arena.i4h_aorta_reach import AortaReachEnv, make_cfg

    def fail(self, sim):
        raise RuntimeError("binding failed")

    with monkeypatch.context() as patch:
        patch.setattr(AortaReachEnv, "_bind_physics", fail)
        cfg, kwargs = make_cfg()
        with pytest.raises(RuntimeError, match="binding failed"):
            AortaReachEnv(cfg, **kwargs)
    assert SimulationContext.instance() is None
    assert not I4hNewtonManager.on_solver and not NewtonManager._per_world_builder_hooks
    instance = make_env(validate_evidence=True)
    try:
        instance.reset()
        instance.step(action(instance))
    finally:
        close_env(instance)


def test_early_terminal_latches_but_final_obs_is_action_boundary(env, monkeypatch):
    asset = env.scene["catheter"]
    boundary = []
    update = asset.update

    def save_boundary(dt):
        update(dt)
        asset.to_torch()
        boundary.append(asset.q.clone())

    monkeypatch.setattr(asset, "update", save_boundary)
    # Timeout on the first sample, with enough action motion to distinguish the boundary.
    wp.to_torch(asset.assessment.elapsed).fill_(120 - env.physics_dt)
    asset.to_warp()
    _, reward, term, trunc, extras = env.step(action(env, 0.2))
    assert trunc.all() and not term.any() and not reward.any()
    final = extras["final_obs"]["policy"]
    assert torch.equal(final[:, :576], boundary[-1].flatten(1))
    rates = torch.tensor([0.5, 1.5, 1.0], device=env.device)
    assert torch.allclose(final[:, -3:], 0.2 * rates * env.step_dt)
    assert torch.allclose(extras["terminal_task"][:, 9], torch.full_like(asset.rows[:, 9], 120))
    assert not asset.targets.any()  # reset occurs after the final observation was copied


def test_mid_action_completion_rewards_once(env):
    asset = env.scene["catheter"]
    # A broad synthetic target qualifies every recorded tip position, independently of navigation.
    asset.assessment.steps[0].fixed = wp.vec3(0, 0, 0)
    asset.assessment.steps[0].radius = 100.0
    wp.to_torch(asset.assessment.active_hold).fill_(0.5 - env.physics_dt)
    asset.to_warp()
    _, reward, term, trunc, extras = env.step(action(env, 0.2))
    assert term.all() and not trunc.any()
    assert torch.allclose(reward, torch.ones_like(reward))
    assert torch.allclose(extras["terminal_task"][:, 9], torch.full_like(reward, env.physics_dt))
    assert torch.allclose(extras["terminal_history"][:, 0, 1], torch.full_like(asset.history[:, 0, 1], env.physics_dt))
    expected = torch.tensor([0.5, 1.5, 1.0], device=env.device) * 0.2 * env.step_dt
    assert torch.allclose(extras["final_obs"]["policy"][:, -3:], expected)


def test_unset_case_queries_and_construction_validation(monkeypatch):
    from types import SimpleNamespace

    from i4h_isaaclab.newton_manager import I4hNewtonManager, I4hSolverCfg, construction_context
    from isaaclab_newton.physics import NewtonManager

    cfg = I4hSolverCfg()
    with monkeypatch.context() as patch:
        patch.setattr(NewtonManager, "_cfg", SimpleNamespace(solver_cfg=cfg))
        # Querying an unconfigured case must not dereference None.
        I4hNewtonManager.handles_decimation()
        I4hNewtonManager.set_decimation(4)
    with pytest.raises(ValueError, match="case must contain"):
        with construction_context(cfg, device="cuda:0", envs=1):
            pytest.fail("an absent case must fail before construction")


def test_failure_after_binding_invalidates_new_owner(monkeypatch):
    from i4h_isaaclab.newton_manager import I4hNewtonManager
    from isaaclab.sim import SimulationContext

    from i4h_arena.i4h_aorta_reach import AortaReachEnv, make_cfg

    original = AortaReachEnv._bind_physics
    owners = []

    def fail_after_bind(self, sim):
        original(self, sim)
        owners.append((sim.physics, self.scene["catheter"].evidence))
        raise RuntimeError("after binding")

    monkeypatch.setattr(AortaReachEnv, "_bind_physics", fail_after_bind)
    cfg, kwargs = make_cfg(decimation=4)
    with pytest.raises(RuntimeError, match="after binding"):
        AortaReachEnv(cfg, **kwargs)
    assert owners and not owners[0][0].valid and not owners[0][1].valid
    assert I4hNewtonManager.simulator is None and not I4hNewtonManager.on_solver
    assert SimulationContext.instance() is None


@pytest.mark.parametrize("schedule", ["physics", "action"])
def test_asset_defers_assessment_until_action_boundary(env, schedule, monkeypatch):
    asset = env.scene["catheter"]
    asset.assessment.steps[0].fixed = wp.vec3(0, 0, 0)
    asset.assessment.steps[0].radius = 100.0
    wp.to_torch(asset.assessment.active_hold).fill_(0.5 - env.step_dt)
    asset.to_warp()
    initial = asset.rows.clone()
    consume = asset.assessment.consume
    calls = []

    def counted(*args, **kwargs):
        calls.append(True)
        return consume(*args, **kwargs)

    monkeypatch.setattr(asset.assessment, "consume", counted)
    steps = 0
    updates = 0

    def update_scene(dt):
        nonlocal steps, updates
        assert dt == pytest.approx(env.physics_dt)
        env._base_scene_update(dt)
        steps += round(dt / env.physics_dt)
        if schedule == "physics":
            updates += 1
            asset.update(dt)
        elif steps == env.cfg.decimation:
            updates += 1
            asset.update(env.step_dt)
        if steps < env.cfg.decimation:
            asset.to_torch()
            assert not calls and torch.equal(asset.rows, initial)
        else:
            assert len(calls) == 1
            before = snapshot(asset)
            asset.update(env.step_dt)
            env.observation_manager.compute()
            assert len(calls) == 1
            assert all(torch.equal(a, b) for a, b in zip(before, snapshot(asset)))

    monkeypatch.setattr(env.scene, "update", update_scene)
    _, reward, term, trunc, extras = env.step(action(env))
    assert steps == env.cfg.decimation and len(calls) == 1
    assert updates == (env.cfg.decimation if schedule == "physics" else 1)
    assert term.all() and not trunc.any() and (reward == 1).all()
    assert (extras["final_obs"]["task"][:, 11] == 1).all()


@pytest.mark.parametrize("fault", ["underfilled", "overflow", "count_only", "overflow_only"])
def test_rejected_evidence_preserves_state_and_reset_recovers(env, fault):
    asset = env.scene["catheter"]
    term = env.action_manager.get_term("catheter")
    term.process_actions(action(env))
    steps = env.cfg.decimation + (1 if fault == "overflow" else -1 if fault == "underfilled" else 0)
    for _ in range(steps):
        term.apply_actions()
        env.sim.step(render=False)
    if fault == "count_only":
        asset.evidence.count.fill_(env.cfg.decimation + 1)
    elif fault == "overflow_only":
        asset.evidence.overflow.fill_(1)
    arrays = [a for a in vars(asset.assessment).values() if isinstance(a, wp.array)]
    before = [a.numpy().copy() for a in arrays]
    exported = [asset.terminal_rows.clone(), asset.terminal_history.clone(), asset.terminal_progress.clone()]
    remaining = asset._remaining_steps
    with pytest.raises(RuntimeError, match=r"env 0: count=.*overflow="):
        asset.update(env.step_dt)
    assert asset._remaining_steps == remaining
    for a, saved in zip(arrays, before):
        assert np.array_equal(a.numpy(), saved)
    assert all(
        torch.equal(a, b)
        for a, b in zip(exported, [asset.terminal_rows, asset.terminal_history, asset.terminal_progress])
    )
    env.reset()
    obs, reward, term, trunc, _ = env.step(action(env))
    assert torch.isfinite(obs["policy"]).all()
    assert not reward.any() and not term.any() and not trunc.any()
    assert np.all(asset.assessment.consumed.numpy() == env.cfg.decimation)
    assert torch.allclose(asset.rows[:, 9], torch.full_like(asset.rows[:, 9], env.step_dt))

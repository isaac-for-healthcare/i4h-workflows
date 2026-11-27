# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Pinned liver BC import, hosted evaluation and isolation/replay qualification."""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def schedule(count, seed):
    rng = np.random.default_rng(seed)
    result, previous = [], -1
    while len(result) < count:
        cycle = rng.permutation(5)
        if cycle[0] == previous:
            cycle[[0, 1]] = cycle[[1, 0]]
        for target in cycle:
            episode_seed = int(rng.integers(0, 2**31 - 1))
            jitter = np.random.default_rng(episode_seed).uniform(-0.005, 0.005, 3)
            result.append(dict(target_index=int(target), seed=episode_seed, jitter_m=jitter.tolist()))
        previous = int(cycle[-1])
    return result[:count]


def prepare_starts(output, episodes):
    if episodes <= 0:
        raise ValueError("evaluation episode count must be positive")
    development = schedule(20, 81273)
    test = schedule(episodes, 981304)
    if {s["seed"] for s in development} & {s["seed"] for s in test}:
        raise ValueError("development and test starts overlap")
    write_json(output / "development.json", development)
    write_json(output / "test.json", test)


def select_checkpoint(candidates, output, starts):
    """Choose earliest maximum development success; never inspect test scores."""
    if not candidates or any(row["episodes"] != 20 for row in candidates):
        raise ValueError("checkpoint selection requires 20 development episodes per candidate")
    best = max(candidates, key=lambda row: (row["successes"], -row["update"]))
    output.parent.mkdir(parents=True, exist_ok=True)
    selected = output.with_name("model_selected.pt")
    shutil.copy2(best["checkpoint"], selected)
    write_json(
        output,
        dict(
            criterion="earliest maximum full-cycle success on 20 development starts",
            starts_sha256=digest(starts),
            candidates=candidates,
            selected_update=best["update"],
            checkpoint=str(selected),
            checkpoint_sha256=digest(selected),
        ),
    )


def evaluate_checkpoints(env, args):
    import torch
    from tensordict import TensorDict

    specs = json.loads(args.starts.read_text())
    metadata = json.loads((args.checkpoint / "run.json").read_text())
    wanted = set(args.candidate_updates)
    rows = []
    for path in metadata["checkpoints"]:
        native = torch.load(path, map_location=env.device, weights_only=False)
        update = native["liver_safeguards"]["completed_updates"]
        if update not in wanted:
            continue
        if any(row["update"] == update for row in rows):
            raise ValueError(f"duplicate checkpoint for update {update}")
        td = TensorDict({"policy": env.liver.observation()}, batch_size=[env.num_envs])
        actor = liver_actor(td).to(env.device).eval()
        actor.load_state_dict(native["actor_state_dict"], strict=True)
        report = evaluate(
            env, specs, lambda values, actor=actor: actor(TensorDict({"policy": values}, batch_size=[len(values)]))
        )
        rows.append(dict(update=update, checkpoint=path, checkpoint_sha256=digest(path), **report))
    if {row["update"] for row in rows} != wanted:
        raise ValueError("missing requested development checkpoint(s)")
    select_checkpoint(rows, args.output, args.starts)


def liver_actor(td):
    """The liver actor: 57 features to 4 action means, matching the RL trainer config."""
    from rsl_rl.models import MLPModel

    return MLPModel(
        td,
        {"actor": ["policy"]},
        "actor",
        4,
        hidden_dims=[128, 128],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=dict(class_name="rsl_rl.modules:GaussianDistribution", init_std=0.02),
    )


def import_bc(checkpoint, observations, output):
    import importlib.metadata

    import torch
    from rsl_rl.models import MLPModel
    from tensordict import TensorDict

    from i4h_arena.medical.liver import ENCODER, normalized_action

    output.mkdir(parents=True, exist_ok=True)
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    obs = torch.as_tensor(np.load(observations)["observations"], dtype=torch.float32)
    td = TensorDict({"policy": obs}, batch_size=[len(obs)])
    actor = liver_actor(td)
    actor.load_state_dict(state["actor"], strict=True)
    actor.eval()
    critic = MLPModel(
        td, {"critic": ["policy"]}, "critic", 1, hidden_dims=[128, 128], activation="elu", obs_normalization=False
    )
    optimizer = torch.optim.Adam(list(actor.parameters()) + list(critic.parameters()), lr=1e-5)
    torch.save(
        dict(
            actor_state_dict=actor.state_dict(),
            critic_state_dict=critic.state_dict(),
            optimizer_state_dict=optimizer.state_dict(),
            iter=0,
            infos={"source_bc_sha256": digest(checkpoint)},
        ),
        output / "native_bc.pt",
    )
    exported = torch.jit.script(actor.as_jit().cpu()).eval()
    exported.save(str(output / "policy.pt"))
    reloaded = torch.jit.load(str(output / "policy.pt"))
    with torch.no_grad():
        expected = actor(td)
        actual = reloaded(obs)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        actor(td, stochastic_output=True)
        mean, std = tuple(p.clone() for p in actor.output_distribution_params)
        torch.testing.assert_close(std, torch.full_like(std, 0.02), rtol=1e-6, atol=0)
        log_prob = actor.get_output_log_prob(expected)
        reference = torch.distributions.Normal(mean, std).log_prob(expected).sum(-1)
        torch.testing.assert_close(log_prob, reference, rtol=0, atol=0)
        kl = actor.get_kl_divergence((mean, std), (actual, std))
        torch.testing.assert_close(kl, torch.zeros_like(kl), rtol=0, atol=0)
        physical = normalized_action(expected)
        physical[:, :3] *= 0.001
        physical[:, 3] = (physical[:, 3] + 1) / 2
    contract = dict(
        objective="full_cycle",
        encoder=ENCODER,
        action_layout=["delta_x", "delta_y", "delta_z", "grip"],
        xyz_vector_limit_m=0.001,
        physics_hz=60,
        control_hz=30,
        decimation=2,
        checkpoint_sha256=digest(checkpoint),
        policy_sha256=digest(output / "policy.pt"),
        frozen_observations_sha256=digest(observations),
        std_type=actor.distribution.std_type,
        std_channels=std[0].tolist(),
        rsl_rl_version=importlib.metadata.version("rsl-rl-lib"),
    )
    write_json(output / "contract.json", contract)
    report = dict(
        mean_max_abs_error=float((actual - expected).abs().max()),
        max_analytic_kl=float(kl.max()),
        log_prob_max_abs_error=float((log_prob - reference).abs().max()),
        physical_xyz_max_m=float(torch.linalg.vector_norm(physical[:, :3], dim=-1).max()),
        grip_min=float(physical[:, 3].min()),
        grip_max=float(physical[:, 3].max()),
        samples=len(obs),
        distribution_std_channels=std[0].tolist(),
    )
    write_json(output / "parity.json", report)
    print(json.dumps(report), flush=True)


def train_bc(demonstrations, output, *, epochs=300, batch_size=512, learning_rate=1e-3, seed=0, holdout=0.1):
    """Fit the liver actor's action means to reference-controller demonstrations.

    Episodes, not samples, are held out so validation measures unseen starts.
    Writes ``bc.pt`` (``{"actor": state_dict}``) and ``observations.npz`` in the
    format ``import_bc`` consumes, plus ``bc_training.json``.
    """
    import torch
    from tensordict import TensorDict

    output.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(seed)
    with np.load(demonstrations) as saved:
        observations = saved["observations"].astype(np.float32)
        actions = saved["actions"].astype(np.float32)
        episodes = saved["episode"]
    if observations.ndim != 2 or observations.shape[1] != 57 or actions.shape != (len(observations), 4):
        raise ValueError(f"demonstrations must be (N, 57) observations and (N, 4) actions: {demonstrations}")
    if not (np.isfinite(observations).all() and np.isfinite(actions).all()):
        raise ValueError("demonstrations contain non-finite values")
    unique = np.unique(episodes)
    rng = np.random.default_rng(seed)
    held = rng.choice(unique, size=max(1, int(round(len(unique) * holdout))), replace=False) if len(unique) > 1 else []
    validation = np.isin(episodes, held)
    if validation.all():
        validation[:] = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    obs = torch.as_tensor(observations, device=device)
    target = torch.as_tensor(actions, device=device)
    actor = liver_actor(TensorDict({"policy": obs[:1]}, batch_size=[1])).to(device)
    optimizer = torch.optim.Adam(actor.parameters(), lr=learning_rate)
    train_ids = torch.as_tensor(np.flatnonzero(~validation), device=device)
    valid_ids = torch.as_tensor(np.flatnonzero(validation), device=device)

    def loss(ids):
        td = TensorDict({"policy": obs[ids]}, batch_size=[len(ids)])
        return torch.nn.functional.mse_loss(actor(td), target[ids])

    history = []
    for epoch in range(epochs):
        actor.train()
        order = train_ids[torch.randperm(len(train_ids), device=device)]
        total = 0.0
        for start in range(0, len(order), batch_size):
            batch = order[start : start + batch_size]
            value = loss(batch)
            optimizer.zero_grad()
            value.backward()
            optimizer.step()
            total += float(value.detach()) * len(batch)
        if epoch == epochs - 1 or epoch % max(1, epochs // 10) == 0:
            actor.eval()
            with torch.no_grad():
                held_out = float(loss(valid_ids)) if len(valid_ids) else None
            history.append(dict(epoch=epoch + 1, train_mse=total / len(order), validation_mse=held_out))
            print("LIVER_BC", json.dumps(history[-1]), flush=True)
    actor = actor.cpu().eval()
    torch.save(dict(actor=actor.state_dict()), output / "bc.pt")
    # A bounded, deterministic sample of the training inputs for import parity.
    frozen = np.sort(rng.choice(len(observations), size=min(4096, len(observations)), replace=False))
    np.savez_compressed(output / "observations.npz", observations=observations[frozen])
    report = dict(
        source=str(demonstrations),
        source_sha256=digest(demonstrations),
        samples=len(observations),
        episodes=len(unique),
        validation_episodes=len(held),
        epochs=epochs,
        batch_size=batch_size,
        learning_rate=learning_rate,
        seed=seed,
        history=history,
        bc_sha256=digest(output / "bc.pt"),
    )
    write_json(output / "bc_training.json", report)
    return report


def collect_demonstrations(env, args):
    """Record the reference controller's observations and actions in every world."""
    specs = schedule(args.episodes, args.seed)
    episodes = []
    report = evaluate(env, specs, demonstrations=episodes, action_noise=args.action_noise, noise_seed=args.seed)
    kept = [e for e in episodes if e["success"] or not args.successful_only]
    if not kept:
        raise RuntimeError("the reference controller produced no usable demonstrations")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        observations=np.concatenate([e["observations"] for e in kept]),
        actions=np.concatenate([e["actions"] for e in kept]),
        episode=np.concatenate([np.full(len(e["actions"]), e["evaluation_id"]) for e in kept]),
    )
    report.update(
        demonstrations=str(args.output.with_suffix(".npz")),
        kept_episodes=len(kept),
        kept_samples=int(sum(len(e["actions"]) for e in kept)),
        action_noise=args.action_noise,
        schedule_seed=args.seed,
    )
    return report


def summarize(rows):
    def group(values):
        n = len(values)
        k = sum(r["success"] for r in values)
        z = 1.959963984540054
        phat = k / n
        den = 1 + z * z / n
        mid = (phat + z * z / (2 * n)) / den
        half = z * ((phat * (1 - phat) / n + z * z / (4 * n * n)) ** 0.5) / den
        return dict(
            episodes=n,
            successes=k,
            success_rate=k / n,
            wilson_95=[mid - half, mid + half],
            failures={
                reason: sum(r["failure_reason"] == reason for r in values)
                for reason in sorted({r["failure_reason"] for r in values if r["failure_reason"]})
            },
        )

    return dict(
        **group(rows),
        per_target={str(t): group([r for r in rows if r["target"] == t]) for t in sorted({r["target"] for r in rows})},
    )


def hosted(args):
    import torch

    from i4h_arena.scenes.base import load_scene
    from i4h_rl.liver_host import build_parser, launch_app

    options = ["--workflow", "liver_reach_retract", "--mode", "idle", "--seed", "7341", "--envs", str(args.worlds)]
    if args.headless:
        options += ["--headless"]
    cli = build_parser().parse_args(options)
    scene = load_scene("liver_reach_retract", cli, register_assets=False)
    scene.configure_args(cli)
    with launch_app(cli) as app:
        env = app.make_env(scene)
        with torch.inference_mode():
            env.reset()
            if args.operation == "prepare-grasped":
                prepare_grasped(env, args.output)
            elif args.operation == "fresh-replay":
                fresh_replay(env, args)
            elif args.operation == "qualify":
                report = qualify(env, args)
                write_json(args.output, report)
            elif args.operation == "collect-demos":
                write_json(args.output, collect_demonstrations(env, args))
            elif args.operation == "select-ppo":
                evaluate_checkpoints(env, args)
            else:
                policy = (
                    None
                    if args.checkpoint is None
                    else torch.jit.load(str(args.checkpoint), map_location=env.device).eval()
                )
                if args.checkpoint:
                    contract = json.loads(args.checkpoint.with_name("contract.json").read_text())
                    if contract["objective"] != env.liver.objective.objective:
                        raise ValueError("policy export and evaluated objective mismatch")
                specs = json.loads(args.starts.read_text()) if args.starts else schedule(args.episodes, 981304)
                report = evaluate(env, specs, policy, trace_path=args.output.with_suffix(".npz"))
                report["starts_sha256"] = (
                    digest(args.starts)
                    if args.starts
                    else hashlib.sha256(json.dumps(specs, sort_keys=True).encode()).hexdigest()
                )
                write_json(args.output, report)
        env.close()


def reset_specs(env, mapping):
    ids = sorted(mapping)
    env.liver.next_specs = mapping
    env.reset(env_ids=torch_ids(env, ids))
    env.liver.next_specs = {}


def torch_ids(env, ids):
    import torch

    return torch.as_tensor(ids, device=env.device, dtype=torch.long)


def evaluate(env, specs, policy=None, trace_path=None, demonstrations=None, action_noise=0.0, noise_seed=0):
    """Run each start once. ``demonstrations`` collects every world's episodes.

    ``action_noise`` perturbs only the executed xyz command; recorded actions stay
    the clean policy or reference output, so the reference labels its own recovery.
    """
    import torch

    generator = torch.Generator(device=env.device).manual_seed(noise_seed)
    buffers = {}

    pending = [iter(enumerate(specs[e :: env.num_envs])) for e in range(env.num_envs)]
    active = {}
    starts = []

    def dispatch(e):
        item = next(pending[e], None)
        if item is None:
            return
        local, spec = item
        active[e] = (e + local * env.num_envs, spec)
        reset_specs(env, {e: spec})
        starts.append(
            dict(
                evaluation_id=active[e][0],
                canonical_row=e,
                tissue_sha256=hashlib.sha256(env.liver.q.numpy()[e].tobytes()).hexdigest(),
                action_sha256=hashlib.sha256(env.liver.physical[e].cpu().numpy().tobytes()).hexdigest(),
                **spec,
            )
        )

    for e in range(env.num_envs):
        dispatch(e)
    rows = []
    trace_observations, trace_actions = [], []
    while active:
        obs = env.liver.observation()
        action = env.liver.teacher_action() if policy is None else policy(obs)
        action = torch.as_tensor(action, device=env.device)
        trace_observations.append(obs[0].cpu().numpy().copy())
        trace_actions.append(action[0].cpu().numpy().copy())
        if demonstrations is not None:
            host_obs, host_action = obs.cpu().numpy(), action.cpu().numpy()
            for e in active:
                buffers.setdefault(e, ([], []))
                buffers[e][0].append(host_obs[e].copy())
                buffers[e][1].append(host_action[e].copy())
        executed = action
        if action_noise:
            executed = action.clone()
            noise = torch.randn(executed[:, :3].shape, device=env.device, generator=generator)
            executed[:, :3] = (executed[:, :3] + action_noise * noise).clamp(-1.0, 1.0)
        _, _, terminated, truncated, extras = env.step(executed)
        for row in extras.get("terminal_results", []):
            e = row["env_id"]
            if e not in active:
                continue
            index, spec = active.pop(e)
            if row["objective"] != env.liver.objective.objective:
                raise ValueError("terminal objective mismatch")
            rows.append(dict(evaluation_id=index, **row))
            if demonstrations is not None:
                observations, actions = buffers.pop(e)
                demonstrations.append(
                    dict(
                        evaluation_id=index,
                        success=bool(row["success"]),
                        observations=np.asarray(observations, dtype=np.float32),
                        actions=np.asarray(actions, dtype=np.float32),
                    )
                )
            print("LIVER_EVALUATION", len(rows), row["target"], row["success"], row["failure_reason"], flush=True)
            dispatch(e)
    if trace_path is not None:
        Path(trace_path).parent.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(trace_path, observations=trace_observations, actions=trace_actions)
    return dict(
        objective=env.liver.objective.objective,
        **summarize(rows),
        results=sorted(rows, key=lambda r: r["evaluation_id"]),
        starts=sorted(starts, key=lambda r: r["evaluation_id"]),
    )


def qualify(env, args):
    import torch

    runtime = env.liver
    env.step(torch.tensor([[0.0, 0.0, 0.0, -1.0]] * env.num_envs, device=env.device))
    binding = env.i4h.physics

    def capture():
        return runtime.capture_state()

    before = capture()
    env.reset(env_ids=torch_ids(env, [0]))
    after = capture()
    for family in ("task", "probe"):
        for key, old in before[family].items():
            np.testing.assert_array_equal(old[1:], after[family][key][1:], err_msg=f"{family}/{key}")
    for owner, key, world, *_ in binding.solver.continuation.entries:
        name = f"{owner.name}:{key}"
        peers = world.numpy() != 0
        np.testing.assert_array_equal(before["numerical"][name][peers], after["numerical"][name][peers], err_msg=name)
    # Exercise ordinary reset independently; only selected IDs may be rewritten.
    env.liver_ordinary_reset = True
    ordinary_before = capture()
    env.reset(env_ids=torch_ids(env, [0]))
    ordinary_after = capture()
    env.liver_ordinary_reset = False
    for family in ("task", "probe"):
        for key, old in ordinary_before[family].items():
            np.testing.assert_array_equal(old[1:], ordinary_after[family][key][1:], err_msg=f"ordinary {family}/{key}")
    for owner, key, world, *_ in binding.solver.continuation.entries:
        name = f"{owner.name}:{key}"
        peers = world.numpy() != 0
        np.testing.assert_array_equal(
            ordinary_before["numerical"][name][peers], ordinary_after["numerical"][name][peers], err_msg=name
        )
    env.reset(env_ids=torch_ids(env, [0]))
    # Action integration is once, including diagonal clipping; both physics samples advance.
    old = runtime.held.clone()
    step = runtime.objective.s[:, 1].clone()
    env.step(torch.tensor([[1.0, 1.0, 1.0, -1.0]] * env.num_envs, device=env.device))
    displacement = torch.linalg.vector_norm(runtime.held[:, :3] - old[:, :3], dim=-1)
    assert float(displacement.max()) <= 0.00100001
    assert torch.all(runtime.objective.s[:, 1] - step == 2)
    unchanged = runtime.objective.s.clone()
    for _ in range(3):
        runtime.observation()
        runtime.control_reward()
    assert torch.equal(unchanged, runtime.objective.s)
    # Same-process continuation identity and subsequent deterministic transition.
    saved = capture()
    action = torch.tensor([[0.0, 0.0, 0.0, -1.0]] * env.num_envs, device=env.device)
    obs = runtime.observation().clone()
    runtime.restore_complete_state(saved, torch_ids(env, list(range(env.num_envs))))
    env.step(action)
    reference = runtime.observation().clone()
    reference_q = runtime.q.numpy().copy()
    runtime.restore_complete_state(saved, torch_ids(env, list(range(env.num_envs))))
    torch.testing.assert_close(runtime.observation(), obs, rtol=0, atol=0)
    env.step(action)
    error = float((runtime.observation() - reference).abs().max())
    physical_error = float(np.linalg.norm(runtime.q.numpy() - reference_q, axis=-1).max())
    assert physical_error <= 1e-5
    # Continuation rebuilds contact caches; the approved numerical tolerance is
    # 10 micrometres. Encode that physical tolerance with the fixed feature scales.
    actual = runtime.observation()
    for columns, tolerance in (
        (slice(0, 3), 0.001),
        (slice(3, 6), 0.0001),
        (slice(6, 12), 0.001),
        (slice(12, 18), 0.03),
        (slice(18, 21), 0.001),
        (slice(21, 28), 1e-6),
        (slice(28, 34), 0.0),
        (slice(34, 36), 0.01),
        (slice(36, 57), 0.0),
    ):
        torch.testing.assert_close(actual[:, columns], reference[:, columns], rtol=0, atol=tolerance)
    # Save an artifact consumed in another process by the replay qualification stage.
    np.savez_compressed(
        args.output.with_suffix(".npz"),
        **{
            f"{family}/{key}": value
            for family, values in saved.items()
            if isinstance(values, dict)
            for key, value in values.items()
        },
        identity=saved["identity"],
        observation=obs.cpu().numpy(),
        next_observation=reference.cpu().numpy(),
        next_q=reference_q,
    )
    return dict(
        worlds=env.num_envs,
        masked_restore_isolation=True,
        ordinary_reset_isolation=True,
        observed_physics_steps=2,
        max_action_displacement_m=float(displacement.max()),
        pure_getters=True,
        replay_max_feature_error=error,
        snapshot_hash=runtime.snapshot.hash,
        replay_max_particle_error_m=physical_error,
    )


def flatten_bundle(bundle):
    return {
        f"{family}/{key}": value
        for family, values in bundle.items()
        if isinstance(values, dict)
        for key, value in values.items()
    } | {"identity": bundle["identity"]}


def load_bundle(path):
    with np.load(path, allow_pickle=False) as saved:
        bundle = {"identity": saved["identity"].copy()}
        for key in saved.files:
            if "/" in key:
                family, name = key.split("/", 1)
                bundle.setdefault(family, {})[name] = saved[key].copy()
        extras = {key: saved[key].copy() for key in saved.files if "/" not in key and key != "identity"}
    return bundle, extras


def prepare_grasped(env, directory):
    import torch

    runtime = env.liver
    directory.mkdir(parents=True, exist_ok=True)
    if runtime.objective.objective != "full_cycle":
        raise ValueError("generate grasped starts with the full-cycle reference objective")
    records = []
    for target in range(5):
        reset_specs(env, {e: dict(target_index=target, jitter_m=[0.0, 0.0, 0.0]) for e in range(env.num_envs)})
        for _ in range(300):
            action = torch.as_tensor(runtime.teacher_action(), device=env.device)
            phase = runtime.objective.s[:, 0]
            # Once a row qualifies, keep its commanded root and closed grip held
            # while the remaining rows finish their half-second qualification.
            ready = (phase >= 3) & (phase <= 4)
            action[ready, :3] = 0.0
            action[ready, 3] = runtime.physical[ready, 7] * 2 - 1
            env.step(action)
            if torch.any(runtime.objective.s[:, 0] >= 7):
                raise RuntimeError("reference failed before a qualified grasped start")
            if torch.all(runtime.objective.s[:, 0] == 3):
                break
        else:
            raise RuntimeError("reference did not qualify every grasped-start world")
        path = directory / f"grasped_{env.num_envs}_{target}.npz"
        if path.exists():
            raise FileExistsError(path)
        bundle = runtime.capture_state()
        np.savez_compressed(path, **flatten_bundle(bundle))
        records.append(
            dict(
                target=runtime.examples[target]["particle_ids"][0],
                file=path.name,
                sha256=digest(path),
                minimum_original_patch_count=int(runtime.views["counts"][:, 1].min()),
                minimum_retained_count=int(runtime.views["counts"][:, 3].min()),
            )
        )
    write_json(
        directory / f"grasped_manifest_{env.num_envs}.json",
        dict(schema=1, worlds=env.num_envs, records=records, canonical_settled_hash=runtime.snapshot.hash),
    )


def fresh_replay(env, args):
    import torch

    source = args.observations
    bundle, extras = load_bundle(source)
    ids = torch_ids(env, list(range(env.num_envs)))
    env.liver.restore_complete_state(bundle, ids)
    np.testing.assert_array_equal(env.liver.observation().cpu().numpy(), extras["observation"])
    env.step(torch.tensor([[0.0, 0.0, 0.0, -1.0]] * env.num_envs, device=env.device))
    error = float(np.abs(env.liver.observation().cpu().numpy() - extras["next_observation"]).max())
    particle_error = float(np.linalg.norm(env.liver.q.numpy() - extras["next_q"], axis=-1).max())
    assert particle_error <= 1e-5
    np.testing.assert_allclose(env.liver.observation().cpu().numpy(), extras["next_observation"], atol=0.03, rtol=0)
    write_json(
        args.output,
        dict(
            worlds=env.num_envs,
            fresh_process=True,
            initial_observation_exact=True,
            max_encoded_transition_error=error,
            max_particle_error_m=particle_error,
            source_sha256=digest(source),
        ),
    )


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument(
        "operation",
        choices=(
            "import-bc",
            "train-bc",
            "collect-demos",
            "evaluate",
            "qualify",
            "prepare-grasped",
            "fresh-replay",
            "prepare-starts",
            "select-ppo",
        ),
    )
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--observations", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--starts", type=Path)
    parser.add_argument("--candidate-updates", type=int, nargs="+", default=[6, 10, 15, 30])
    parser.add_argument("--episodes", type=int, default=100)
    parser.add_argument("--worlds", type=int, choices=(1, 8, 16), default=16)
    parser.add_argument("--headless", action="store_true")
    parser.add_argument("--seed", type=int, default=20261123, help="collect-demos start schedule and noise seed")
    parser.add_argument("--action-noise", type=float, default=0.0, help="collect-demos executed xyz noise std")
    parser.add_argument("--successful-only", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--demonstrations", type=Path, help="train-bc input from collect-demos")
    parser.add_argument("--epochs", type=int, default=300, help="train-bc epochs")
    args = parser.parse_args()
    if args.operation == "prepare-starts":
        prepare_starts(args.output, args.episodes)
    elif args.operation == "import-bc":
        import_bc(args.checkpoint, args.observations, args.output)
    elif args.operation == "train-bc":
        train_bc(args.demonstrations, args.output, epochs=args.epochs, seed=args.seed)
    else:
        hosted(args)


if __name__ == "__main__":
    main()

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RSL-RL backend for compact policies trained with the simulator runtime."""

from __future__ import annotations

import argparse
import os
import shlex
import shutil
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path

import yaml

from i4h_rl.artifacts import checkpoint_iteration, resolve_input_path, resolve_output_path, resolve_run_dir, write_json
from i4h_rl.profile import RLProfile


def validate_profile(profile: RLProfile, _workflows_root: Path) -> None:
    """Validate the lightweight portion of a declarative RSL-RL configuration."""
    try:
        config = yaml.safe_load(profile.trainer_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read RSL-RL config {profile.trainer_config}: {exc}") from exc
    if not isinstance(config, dict) or config.get("schema_version") != 1 or config.get("backend") != "rsl_rl":
        raise SystemExit(f"{profile.trainer_config}: expected schema_version 1 and backend rsl_rl")
    try:
        runner = config["runner"]
        policy = config["policy"]
        algorithm = config["algorithm"]
        experiment_name = runner["experiment_name"]
        actor_groups = runner["obs_groups"]["actor"]
        critic_groups = runner["obs_groups"]["critic"]
        runner_counts = {
            "num_steps_per_env": runner["num_steps_per_env"],
            "max_iterations": runner["max_iterations"],
            "save_interval": runner["save_interval"],
        }
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"{profile.trainer_config}: incomplete runner, policy, or algorithm configuration") from exc
    if not isinstance(policy, dict) or not policy or not isinstance(algorithm, dict) or not algorithm:
        raise SystemExit(f"{profile.trainer_config}: policy and algorithm must be non-empty mappings")
    if experiment_name != profile.workflow:
        raise SystemExit(
            f"{profile.trainer_config}: runner.experiment_name={experiment_name!r} "
            f"does not match workflow {profile.workflow!r}"
        )
    if (
        actor_groups != critic_groups
        or not isinstance(actor_groups, list)
        or not actor_groups
        or any(not isinstance(group, str) or not group for group in actor_groups)
    ):
        raise SystemExit(
            f"{profile.trainer_config}: actor and critic observation groups must be equal, non-empty string lists"
        )
    if any(isinstance(value, bool) or not isinstance(value, int) or value < 1 for value in runner_counts.values()):
        raise SystemExit(f"{profile.trainer_config}: RSL-RL runner counts must be positive integers")
    if runner_counts["max_iterations"] != profile.default_epochs:
        raise SystemExit(
            f"{profile.trainer_config}: runner.max_iterations={runner_counts['max_iterations']!r} "
            f"does not match profile default_epochs={profile.default_epochs}"
        )


def _runtime_python(workflows_root: Path, explicit: str | None) -> Path:
    candidates = (
        explicit,
        os.environ.get("I4H_RL_PYTHON"),
        "/isaac-sim/python.sh" if Path("/isaac-sim/python.sh").exists() else None,
        str(workflows_root / "arena/.venv/bin/python"),
    )
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path if path.is_absolute() else (Path.cwd() / path).absolute()
    raise SystemExit(
        "no RSL-RL runtime Python found; set I4H_RL_PYTHON to an environment containing "
        "Isaac Sim, Isaac Lab, IsaacLab-Arena, and RSL-RL"
    )


def _runtime_env(workflows_root: Path, profile: RLProfile) -> dict[str, str]:
    roots = (
        workflows_root / "rl",
        workflows_root / "arena",
        workflows_root / "common",
        workflows_root / "engine",
        workflows_root / "workflows",
    )
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_RUN_RECURSION_DEPTH", None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in roots), *([existing] if existing else [])])
    env["I4H_WORKFLOWS"] = str(workflows_root)
    env["I4H_RL_TRAINER_CONFIG"] = str(profile.trainer_config)
    # Match the normal Workflow launcher. RSL-RL starts Kit through Isaac Lab's
    # trainer entry point instead of run.sh, so it must carry the noninteractive
    # EULA acceptance itself or stdin-less agent/CI launches stop at a prompt.
    if not env.get("OMNI_KIT_ACCEPT_EULA"):
        env["OMNI_KIT_ACCEPT_EULA"] = "YES"
    return env


def _preflight(runtime: Path, env: dict[str, str]) -> None:
    probe = "import i4h_arena, isaaclab, isaaclab_arena, isaaclab_rl, rsl_rl; print('RSL-RL runtime imports: OK')"
    result = subprocess.run([str(runtime), "-c", probe], env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        last = detail[-1] if detail else f"exit {result.returncode}"
        raise SystemExit(f"RSL-RL runtime preflight failed with {runtime}: {last}")
    print(result.stdout.strip())


def _evaluation_command(
    runtime: Path,
    profile: RLProfile,
    checkpoint: Path,
    *,
    episodes: int,
    num_envs: int,
    output: Path,
    export_policy: Path | None = None,
) -> list[str]:
    command = [
        str(runtime),
        "-m",
        "i4h_rl.rsl_rl_eval",
        "--task",
        profile.eval_task_id,
        "--checkpoint",
        str(checkpoint),
        "--episodes",
        str(episodes),
        "--num_envs",
        str(num_envs),
        "--env_spacing",
        str(profile.simulation.env_spacing),
        "--presets",
        profile.simulation.presets,
        "--output",
        str(output),
    ]
    if profile.adapter_module:
        command.extend(("--adapter-module", profile.adapter_module))
    if export_policy is not None:
        command.extend(("--export-policy", str(export_policy)))
    return command


def validate_launch(args: argparse.Namespace, _profile: RLProfile, _workflows_root: Path) -> None:
    if args.sim_runtime_python:
        raise SystemExit("--sim-runtime-python is only valid for RLinf workflows")
    if args.model_path:
        raise SystemExit("RSL-RL trains from scratch; do not pass --model-path")
    if args.resume_dir:
        raise SystemExit("RSL-RL resume is not yet supported by this launcher")
    if args.only_eval and not args.rl_model_path:
        raise SystemExit("RSL-RL evaluation requires --checkpoint")
    if args.only_eval and args.video:
        raise SystemExit("RSL-RL evaluation video is not yet supported; use normal Workflow validation for review")
    if args.only_eval and args.overrides:
        raise SystemExit("RSL-RL evaluation does not accept --set overrides")
    if not args.only_eval and args.rl_model_path:
        raise SystemExit("--checkpoint is only valid with --eval or export")


def launch(
    args: argparse.Namespace,
    profile: RLProfile,
    workflows_root: Path,
    *,
    num_envs: int,
    epochs: int,
    overrides: tuple[str, ...],
) -> int:
    checkpoint: Path | None = None
    if args.only_eval:
        checkpoint = resolve_input_path(workflows_root, args.rl_model_path)
        if not checkpoint.is_file():
            raise SystemExit(f"checkpoint does not exist: {checkpoint}")
    runtime = _runtime_python(workflows_root, args.runtime_python)
    env = _runtime_env(workflows_root, profile)
    _preflight(runtime, env)
    run_dir = resolve_run_dir(workflows_root, profile.workflow, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)

    metadata = {
        "schema_version": 1,
        "workflow": profile.workflow,
        "scene": profile.scene,
        "mode": "rl-eval" if args.only_eval else "rl-train",
        "trainer": profile.trainer,
        "algorithm": profile.algorithm,
        "run_dir": str(run_dir),
        "num_envs": min(num_envs, args.episodes) if args.only_eval else num_envs,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if args.only_eval:
        metadata.update({"checkpoint": str(checkpoint), "episodes": args.episodes})
    else:
        metadata["max_epochs"] = epochs
    write_json(run_dir / "run.json", metadata)

    if args.only_eval:
        assert checkpoint is not None
        evaluation = run_dir / "evaluation.json"
        command = _evaluation_command(
            runtime,
            profile,
            checkpoint,
            episodes=args.episodes,
            num_envs=min(num_envs, args.episodes),
            output=evaluation,
        )
        print(f"run dir: {run_dir}")
        print("launch: " + shlex.join(command), flush=True)
        return_code = subprocess.call(command, env=env, cwd=workflows_root)
        if return_code != 0 or not evaluation.is_file() or evaluation.stat().st_size == 0:
            print(
                f"RSL-RL evaluation did not produce a result (exit={return_code}, result={evaluation})",
                file=sys.stderr,
            )
            return return_code or 1
        return 0

    train_script = workflows_root / "third_party/IsaacLab-ffff603/scripts/reinforcement_learning/rsl_rl/train.py"
    command = [
        str(runtime),
        str(train_script),
        "--external_callback",
        "i4h_rl.rsl_rl_interop.environment_registration_callback",
        "--task",
        profile.train_task_id,
        "--rl_training_mode",
        "--num_envs",
        str(num_envs),
        "--max_iterations",
        str(epochs),
        "--env_spacing",
        str(profile.simulation.env_spacing),
        "--presets",
        profile.simulation.presets,
        f"agent.experiment_name={run_dir}",
    ]
    if args.video:
        command.append("--video")
    command.extend(overrides)
    print(f"run dir: {run_dir}")
    print("launch: " + shlex.join(command), flush=True)
    return_code = subprocess.call(command, env=env, cwd=workflows_root)
    checkpoints = sorted(run_dir.rglob("model_*.pt"), key=checkpoint_iteration)
    parameter_files = sorted(run_dir.rglob("params/agent.yaml"))
    if return_code != 0 or not checkpoints or not parameter_files:
        print(
            "RSL-RL training did not produce a complete checkpoint bundle "
            f"(exit={return_code}, checkpoints={len(checkpoints)}, params={len(parameter_files)})",
            file=sys.stderr,
        )
        return return_code or 1
    final_checkpoint = run_dir / "model_final.pt"
    final_agent_config = run_dir / "agent.yaml"
    shutil.copy2(checkpoints[-1], final_checkpoint)
    shutil.copy2(parameter_files[-1], final_agent_config)
    metadata["checkpoints"] = [str(path) for path in checkpoints]
    metadata["agent_configs"] = [str(path) for path in parameter_files]
    metadata["final_checkpoint"] = str(final_checkpoint)
    metadata["final_agent_config"] = str(final_agent_config)
    write_json(run_dir / "run.json", metadata)
    print(f"training complete: {final_checkpoint}")
    return 0


def export(args: argparse.Namespace, profile: RLProfile, workflows_root: Path) -> int:
    if args.train_config:
        raise SystemExit("--train-config is only valid when exporting an RLinf checkpoint")
    checkpoint = resolve_input_path(workflows_root, args.rl_model_path)
    if not checkpoint.is_file():
        raise SystemExit(f"RSL-RL checkpoint does not exist: {checkpoint}")
    output = resolve_output_path(workflows_root, args.output_dir)
    runtime = _runtime_python(workflows_root, args.runtime_python)
    env = _runtime_env(workflows_root, profile)
    _preflight(runtime, env)
    output.mkdir(parents=True, exist_ok=True)
    policy = output / "policy.pt"
    evaluation = output / "evaluation.json"
    command = _evaluation_command(
        runtime,
        profile,
        checkpoint,
        episodes=1,
        num_envs=1,
        output=evaluation,
        export_policy=policy,
    )
    print(f"workflow: {profile.workflow}")
    print(f"RSL-RL checkpoint: {checkpoint}")
    print(f"TorchScript policy: {policy}")
    print("launch: " + shlex.join(command), flush=True)
    return_code = subprocess.call(command, env=env, cwd=workflows_root)
    if return_code != 0 or not policy.is_file() or not evaluation.is_file():
        print(
            "RSL-RL export did not produce a complete policy bundle "
            f"(exit={return_code}, policy={policy.is_file()}, evaluation={evaluation.is_file()})",
            file=sys.stderr,
        )
        return return_code or 1
    manifest = {
        "schema_version": 1,
        "workflow": profile.workflow,
        "scene": profile.scene,
        "trainer": profile.trainer,
        "source_checkpoint": str(checkpoint),
        "policy": str(policy),
        "state_dof": profile.state_dof,
        "action_dof": profile.action_dof,
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(output / "policy.json", manifest)
    print(f"export complete: {policy}")
    return 0

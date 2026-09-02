# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""RLinf backend for foundation-policy post-training with an isolated simulator."""

from __future__ import annotations

import argparse
import json
import math
import os
import secrets
import shlex
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

import yaml

from i4h_rl.artifacts import resolve_input_path, resolve_output_path, resolve_run_dir, write_json
from i4h_rl.profile import RLProfile


def _config(profile: RLProfile) -> dict[str, object]:
    try:
        raw = yaml.safe_load(profile.trainer_config.read_text(encoding="utf-8"))
    except (OSError, yaml.YAMLError) as exc:
        raise SystemExit(f"cannot read RLinf config {profile.trainer_config}: {exc}") from exc
    if not isinstance(raw, dict):
        raise SystemExit(f"{profile.trainer_config}: expected a YAML mapping")
    return raw


def validate_profile(profile: RLProfile, _workflows_root: Path) -> None:
    """Verify profile facts duplicated in the RLinf/Hydra configuration."""
    config = _config(profile)
    try:
        train = config["env"]["train"]
        evaluate = config["env"]["eval"]
        train_init = train["init_params"]
        eval_init = evaluate["init_params"]
        action_dim = config["actor"]["model"]["action_dim"]
        train_action_mapping = train["isaaclab"]["action_mapping"]
        eval_action_mapping = evaluate["isaaclab"]["action_mapping"]
        train_episode_steps = train["max_episode_steps"]
        eval_episode_steps = evaluate["max_episode_steps"]
    except (KeyError, TypeError) as exc:
        raise SystemExit(f"{profile.trainer_config}: incomplete RLinf environment or actor configuration") from exc
    if not all(isinstance(value, dict) for value in (train_init, eval_init)):
        raise SystemExit(f"{profile.trainer_config}: RLinf train/eval init_params must be mappings")

    expected = (
        ("env.train.init_params.id", train_init.get("id"), profile.train_task_id),
        ("env.eval.init_params.id", eval_init.get("id"), profile.eval_task_id),
        (
            "env.train.init_params.task_description",
            train_init.get("task_description"),
            profile.task_description,
        ),
        (
            "env.eval.init_params.task_description",
            eval_init.get("task_description"),
            profile.task_description,
        ),
        ("actor.model.action_dim", action_dim, profile.policy_action_dof),
    )
    for key, actual, declared in expected:
        if actual != declared:
            raise SystemExit(f"{profile.trainer_config}: {key}={actual!r} does not match profile value {declared!r}")

    for section, action_mapping in (("train", train_action_mapping), ("eval", eval_action_mapping)):
        try:
            mapped_dof = (
                int(action_mapping.get("prefix_pad", 0))
                + profile.policy_action_dof
                + int(action_mapping.get("suffix_pad", 0))
            )
        except (AttributeError, TypeError, ValueError) as exc:
            raise SystemExit(
                f"{profile.trainer_config}: env.{section} action mapping padding must be integers"
            ) from exc
        if mapped_dof != profile.action_dof:
            raise SystemExit(
                f"{profile.trainer_config}: env.{section} mapped action dimension {mapped_dof} does not match "
                f"profile action_dof={profile.action_dof}"
            )

    try:
        episode_steps = (int(train_episode_steps), int(eval_episode_steps))
    except (TypeError, ValueError) as exc:
        raise SystemExit(f"{profile.trainer_config}: train/eval max_episode_steps must be integers") from exc
    if any(value < 1 for value in episode_steps) or episode_steps[0] != episode_steps[1]:
        raise SystemExit(f"{profile.trainer_config}: train/eval max_episode_steps must be equal and positive")


def _model_runtime_python(workflows_root: Path, explicit: str | None) -> Path:
    candidates = (
        explicit,
        os.environ.get("I4H_RL_PYTHON"),
        str(workflows_root / "tasks/gr00t_n15/.venv/bin/python"),
    )
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path if path.is_absolute() else (Path.cwd() / path).absolute()
    raise SystemExit("no RLinf model runtime found; run setup.sh tasks/gr00t_n15 or set I4H_RL_PYTHON")


def _sim_runtime_python(workflows_root: Path, explicit: str | None) -> Path:
    candidates = (
        explicit,
        os.environ.get("I4H_RL_SIM_PYTHON"),
        str(workflows_root / "arena/.venv/bin/python"),
    )
    for candidate in candidates:
        if candidate:
            path = Path(candidate).expanduser()
            if path.is_file():
                return path if path.is_absolute() else (Path.cwd() / path).absolute()
    raise SystemExit("no RLinf simulator runtime found; run setup.sh arena or set I4H_RL_SIM_PYTHON")


def _runtime_env(workflows_root: Path, profile: RLProfile) -> dict[str, str]:
    third_party = workflows_root / "third_party"
    rlinf_dirs = sorted(third_party.glob("RLinf-*"))
    if not rlinf_dirs:
        raise SystemExit("RLinf checkout is missing; run ./third_party/setup.sh")
    roots = (
        workflows_root / "rl",
        workflows_root / "common",
        workflows_root / "tasks/gr00t_n15",
        rlinf_dirs[-1],
        third_party / "Isaac-GR00T-1.5",
        third_party / "IsaacLab-ffff603/source/isaaclab_contrib",
    )
    env = os.environ.copy()
    env.pop("VIRTUAL_ENV", None)
    env.pop("UV_RUN_RECURSION_DEPTH", None)
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = os.pathsep.join([*(str(path) for path in roots), *([existing] if existing else [])])
    env["I4H_WORKFLOWS"] = str(workflows_root)
    env["I4H_RL_ADAPTER_MODULE"] = profile.adapter_module or ""
    env["RLINF_EXT_MODULE"] = "i4h_rl.extension"
    env["RLINF_CONFIG_FILE"] = str(profile.trainer_config)
    env["REPO_PATH"] = str(rlinf_dirs[-1])
    env["RAY_ENABLE_UV_RUN_RUNTIME_ENV"] = "0"
    return env


def _sim_env(workflows_root: Path, model_env: dict[str, str], *, gpu: str) -> dict[str, str]:
    sim_roots = (
        workflows_root / "rl",
        workflows_root / "arena",
        workflows_root / "common",
        workflows_root / "engine",
        workflows_root / "workflows",
    )
    env = model_env.copy()
    env["PYTHONPATH"] = os.pathsep.join(str(path) for path in sim_roots)
    env["CUDA_VISIBLE_DEVICES"] = gpu
    return env


def _gpu_assignment() -> tuple[str, str]:
    model_gpu = os.environ.get("I4H_RL_MODEL_GPU", "0").strip()
    sim_gpu = os.environ.get("I4H_RL_SIM_GPU", "1").strip()
    if not model_gpu or not sim_gpu:
        raise SystemExit("I4H_RL_MODEL_GPU and I4H_RL_SIM_GPU must name visible physical GPUs")
    if model_gpu == sim_gpu:
        raise SystemExit("RLinf model and simulator processes require distinct GPUs")
    return model_gpu, sim_gpu


def _sim_ready_timeout() -> float:
    raw = os.environ.get("I4H_RL_SIM_READY_TIMEOUT_S", "180")
    try:
        timeout = float(raw)
    except ValueError as exc:
        raise SystemExit("I4H_RL_SIM_READY_TIMEOUT_S must be a positive number") from exc
    if not math.isfinite(timeout) or timeout <= 0:
        raise SystemExit("I4H_RL_SIM_READY_TIMEOUT_S must be a positive number")
    return timeout


def _model_preflight(runtime: Path, env: dict[str, str], *, require_two_gpus: bool) -> None:
    gpu_check = (
        "assert torch.cuda.device_count() >= 2, 'RLinf training requires two visible GPUs'; "
        if require_two_gpus
        else ""
    )
    probe = (
        "import gr00t, ray, rlinf, torch; "
        "assert tuple(map(int, ray.__version__.split('.')[:2])) >= (2, 47); "
        f"{gpu_check}"
        "from i4h_rl import extension; extension.register(); "
        "print('RLinf model runtime imports: OK')"
    )
    result = subprocess.run([str(runtime), "-c", probe], env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        last = detail[-1] if detail else f"exit {result.returncode}"
        raise SystemExit(f"RLinf model runtime preflight failed with {runtime}: {last}")
    print(result.stdout.strip())


def _sim_preflight(sim_runtime: Path, env: dict[str, str]) -> None:
    probe = (
        "import i4h_arena, isaaclab, isaaclab_arena; "
        "from i4h_rl import sim_server; "
        "print('RLinf simulator runtime imports: OK')"
    )
    result = subprocess.run([str(sim_runtime), "-c", probe], env=env, text=True, capture_output=True, check=False)
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        last = detail[-1] if detail else f"exit {result.returncode}"
        raise SystemExit(f"RLinf simulator runtime preflight failed with {sim_runtime}: {last}")
    print(result.stdout.strip())


def _episode_steps(profile: RLProfile, *, only_eval: bool, overrides: tuple[str, ...]) -> int:
    section = "eval" if only_eval else "train"
    raw = _config(profile)
    try:
        value = int(raw["env"][section]["max_episode_steps"])
    except (KeyError, TypeError, ValueError) as exc:
        raise SystemExit(f"{profile.trainer_config}: env.{section}.max_episode_steps must be an integer") from exc
    override_key = f"env.{section}.max_episode_steps"
    for override in overrides:
        key, _, setting = override.partition("=")
        if key == override_key:
            try:
                value = int(setting)
            except ValueError as exc:
                raise SystemExit(f"{override_key} must be an integer") from exc
    if value < 1:
        raise SystemExit(f"{override_key} must be positive")
    return value


def weights(path: Path) -> Path:
    payload = checkpoint_bundle(path)
    if payload is not None:
        bundle = path / "checkpoint.json"
        declared = payload["weights"]
        assert isinstance(declared, str)
        path = Path(declared).expanduser()
        if not path.is_absolute():
            path = (bundle.parent / path).resolve()
    candidates = (
        path,
        path / "full_weights.pt",
        path / "model_state_dict/full_weights.pt",
        path / "actor/model_state_dict/full_weights.pt",
    )
    for candidate in candidates:
        if candidate.is_file() and candidate.name == "full_weights.pt":
            return candidate.resolve()
    raise SystemExit(
        f"cannot find RLinf actor weights under {path}; expected full_weights.pt, "
        "model_state_dict/full_weights.pt, or actor/model_state_dict/full_weights.pt"
    )


def checkpoint_root(actor_weights: Path) -> Path:
    if actor_weights.parent.name == "model_state_dict" and actor_weights.parent.parent.name == "actor":
        return actor_weights.parent.parent.parent
    raise SystemExit(
        "RLinf evaluation requires the native checkpoint layout "
        f"<checkpoint>/actor/model_state_dict/full_weights.pt; got {actor_weights}"
    )


def train_config(checkpoint_source: Path, actor_weights: Path) -> Path | None:
    search_roots = [checkpoint_source] if checkpoint_source.is_dir() else []
    search_roots.extend([actor_weights.parent, *tuple(actor_weights.parents)[:10]])
    seen: set[Path] = set()
    for directory in search_roots:
        if directory in seen:
            continue
        seen.add(directory)
        for candidate in (directory / "tensorboard/config.yaml", directory / "config.yaml"):
            if candidate.is_file():
                try:
                    payload = yaml.safe_load(candidate.read_text(encoding="utf-8"))
                except (OSError, yaml.YAMLError):
                    continue
                if isinstance(payload, dict) and isinstance(payload.get("actor"), dict):
                    return candidate.resolve()
    return None


def checkpoint_bundle(path: Path, *, expected_workflow: str | None = None) -> dict[str, object] | None:
    bundle = path / "checkpoint.json" if path.is_dir() else None
    if bundle is None or not bundle.is_file():
        return None
    try:
        payload = json.loads(bundle.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit(f"cannot read RL checkpoint bundle {bundle}: {exc}") from exc
    if not isinstance(payload, dict):
        raise SystemExit(f"{bundle}: expected a JSON object")
    if payload.get("schema_version") != 1 or payload.get("format") != "rlinf-fsdp":
        raise SystemExit(f"{bundle}: expected schema_version 1 and format rlinf-fsdp")
    required_paths = ("checkpoint", "weights", "base_model", "trainer_config")
    for key in required_paths:
        value = payload.get(key)
        if not isinstance(value, str) or not value.strip():
            raise SystemExit(f"{bundle}: {key} must be a non-empty path")
    workflow = payload.get("workflow")
    if not isinstance(workflow, str) or not workflow.strip():
        raise SystemExit(f"{bundle}: workflow must be a non-empty string")
    if expected_workflow is not None and workflow != expected_workflow:
        raise SystemExit(f"{bundle}: workflow {workflow!r} does not match selected workflow {expected_workflow!r}")
    return payload


def finalize_training(run_dir: Path, metadata: dict[str, object]) -> bool:
    candidates = list(run_dir.rglob("actor/model_state_dict/full_weights.pt"))
    if not candidates:
        return False

    def global_step(path: Path) -> int:
        for part in reversed(path.parts):
            if part.startswith("global_step_"):
                try:
                    return int(part.removeprefix("global_step_"))
                except ValueError:
                    break
        return -1

    actor_weights = max(candidates, key=lambda path: (global_step(path), path.stat().st_mtime_ns))
    checkpoint = checkpoint_root(actor_weights)
    bundle = {
        "schema_version": 1,
        "format": "rlinf-fsdp",
        "workflow": metadata["workflow"],
        "checkpoint": str(checkpoint.relative_to(run_dir)),
        "weights": str(actor_weights.relative_to(run_dir)),
        "base_model": metadata["model_path"],
        "trainer_config": metadata["trainer_config"],
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(run_dir / "checkpoint.json", bundle)
    metadata["final_checkpoint"] = str(checkpoint)
    metadata["final_weights"] = str(actor_weights)
    metadata["checkpoint_bundle"] = str(run_dir / "checkpoint.json")
    write_json(run_dir / "run.json", metadata)
    print(f"training complete: {run_dir}")
    print(f"checkpoint bundle: {run_dir / 'checkpoint.json'}")
    return True


def finalize_evaluation(
    run_dir: Path,
    metadata: dict[str, object],
    *,
    runtime: Path,
    env: dict[str, str],
) -> bool:
    event_dir = run_dir / "tensorboard"
    events = sorted(event_dir.glob("events.out.tfevents.*"))
    if not events or not any(path.stat().st_size > 0 for path in events):
        return False
    probe = """
import json
import sys
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

accumulator = EventAccumulator(sys.argv[1]).Reload()
metrics = {
    tag: accumulator.Scalars(tag)[-1].value
    for tag in accumulator.Tags().get("scalars", [])
    if accumulator.Scalars(tag)
}
print(json.dumps(metrics, sort_keys=True))
"""
    result = subprocess.run(
        [str(runtime), "-c", probe, str(event_dir)],
        env=env,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode:
        detail = (result.stderr or result.stdout).strip().splitlines()
        last = detail[-1] if detail else f"exit {result.returncode}"
        print(f"cannot read RLinf evaluation metrics: {last}", file=sys.stderr)
        return False
    try:
        metrics = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        print(f"cannot decode RLinf evaluation metrics: {exc}", file=sys.stderr)
        return False
    required = {"eval/return", "eval/success_once", "eval/episode_len", "eval/num_trajectories"}
    if not isinstance(metrics, dict) or not required.issubset(metrics):
        print(
            f"RLinf evaluation metrics are incomplete; expected {sorted(required)}, got {sorted(metrics)}",
            file=sys.stderr,
        )
        return False
    try:
        numeric_metrics = {key: float(metrics[key]) for key in required}
    except (TypeError, ValueError):
        print("RLinf evaluation metrics must be numeric", file=sys.stderr)
        return False
    if not all(math.isfinite(value) for value in numeric_metrics.values()):
        print("RLinf evaluation metrics must be finite", file=sys.stderr)
        return False
    if numeric_metrics["eval/num_trajectories"] <= 0:
        print("RLinf evaluation produced no trajectories", file=sys.stderr)
        return False
    evaluation = {
        "schema_version": 1,
        "workflow": metadata["workflow"],
        "scene": metadata["scene"],
        "trainer": metadata["trainer"],
        "checkpoint": metadata["native_checkpoint"],
        "metrics": metrics,
        "created_at": datetime.now(UTC).isoformat(),
    }
    evaluation_path = run_dir / "evaluation.json"
    write_json(evaluation_path, evaluation)
    metadata["evaluation"] = str(evaluation_path)
    metadata["metrics"] = metrics
    write_json(run_dir / "run.json", metadata)
    print(f"evaluation complete: {evaluation_path}")
    if numeric_metrics["eval/success_once"] <= 0:
        print("RLinf evaluation completed without simulator task success", file=sys.stderr)
        return False
    return True


def validate_launch(args: argparse.Namespace, profile: RLProfile, workflows_root: Path) -> None:
    bundle: dict[str, object] | None = None
    if args.only_eval:
        if not args.rl_model_path:
            raise SystemExit("RLinf evaluation requires --checkpoint")
        checkpoint_source = resolve_input_path(workflows_root, args.rl_model_path)
        bundle = checkpoint_bundle(checkpoint_source, expected_workflow=profile.workflow)
        checkpoint_root(weights(checkpoint_source))
        if args.resume_dir:
            raise SystemExit("RLinf evaluation does not accept --resume-dir")
    elif args.rl_model_path:
        raise SystemExit("--checkpoint is only valid with --eval or export")
    model_value = args.model_path
    if model_value is None and bundle is not None:
        bundled_model = bundle.get("base_model")
        if isinstance(bundled_model, str) and bundled_model:
            model_value = bundled_model
    if not model_value:
        raise SystemExit(
            "--model-path is required for RLinf training; evaluation may omit it when "
            "--checkpoint points to an i4h run bundle"
        )
    model_path = resolve_input_path(workflows_root, model_value)
    if not model_path.exists():
        raise SystemExit(f"--model-path does not exist: {model_path}")
    args.resolved_model_path = model_path
    print(f"model: {model_path}")


def launch(
    args: argparse.Namespace,
    profile: RLProfile,
    workflows_root: Path,
    *,
    num_envs: int,
    epochs: int,
    overrides: tuple[str, ...],
) -> int:
    model_path = args.resolved_model_path
    native_checkpoint: Path | None = None
    if args.only_eval:
        checkpoint_source = resolve_input_path(workflows_root, args.rl_model_path)
        native_checkpoint = checkpoint_root(weights(checkpoint_source))

    model_runtime = _model_runtime_python(workflows_root, args.runtime_python)
    sim_runtime = _sim_runtime_python(workflows_root, args.sim_runtime_python)
    model_env = _runtime_env(workflows_root, profile)
    model_gpu, sim_gpu = _gpu_assignment()
    _model_preflight(model_runtime, model_env, require_two_gpus=True)
    sim_env = _sim_env(workflows_root, model_env, gpu=sim_gpu)
    _sim_preflight(sim_runtime, sim_env)

    run_dir = resolve_run_dir(workflows_root, profile.workflow, args.run_dir)
    run_dir.mkdir(parents=True, exist_ok=False)
    metadata = {
        "schema_version": 1,
        "workflow": profile.workflow,
        "scene": profile.scene,
        "mode": "rl-eval" if args.only_eval else "rl-train",
        "trainer": profile.trainer,
        "algorithm": profile.algorithm,
        "adapter_module": profile.adapter_module,
        "model_path": str(model_path),
        "trainer_config": str(profile.trainer_config),
        "run_dir": str(run_dir),
        "num_envs": num_envs,
        "max_epochs": epochs,
        "model_runtime": str(model_runtime),
        "simulator_runtime": str(sim_runtime),
        "model_gpu": model_gpu,
        "simulator_gpu": sim_gpu,
        "created_at": datetime.now(UTC).isoformat(),
    }
    if native_checkpoint is not None:
        metadata["native_checkpoint"] = str(native_checkpoint)
    write_json(run_dir / "run.json", metadata)

    command = [
        str(model_runtime),
        "-m",
        "i4h_rl.launcher",
        "--config",
        str(profile.trainer_config),
        "--model-path",
        str(model_path),
        "--run-dir",
        str(run_dir),
        "--num-envs",
        str(num_envs),
        "--max-epochs",
        str(epochs),
    ]
    if args.only_eval:
        command.append("--only-eval")
    if args.video:
        command.append("--video")
    if args.resume_dir:
        resume_source = resolve_input_path(workflows_root, args.resume_dir)
        command.extend(("--resume-dir", str(checkpoint_root(weights(resume_source)))))
    if native_checkpoint is not None:
        command.extend(("--rl-model-path", str(native_checkpoint)))
    for override in overrides:
        command.extend(("--set", override))

    socket_path = Path("/tmp") / f"i4h-rl-{os.getpid()}.sock"
    ready_file = run_dir / "simulator.ready"
    ready_file.unlink(missing_ok=True)
    sim_command = [
        str(sim_runtime),
        "-m",
        "i4h_rl.sim_server",
        "--scene",
        profile.scene,
        "--socket",
        str(socket_path),
        "--ready-file",
        str(ready_file),
        "--num-envs",
        str(num_envs),
        "--max-episode-steps",
        str(_episode_steps(profile, only_eval=args.only_eval, overrides=overrides)),
        "--env-spacing",
        str(profile.simulation.env_spacing),
        "--presets",
        profile.simulation.presets,
        "--enable-cameras" if profile.simulation.enable_cameras else "--no-enable-cameras",
    ]
    bridge_key = secrets.token_hex(32)
    sim_env["I4H_RL_SIM_AUTHKEY"] = bridge_key
    model_env["I4H_RL_SIM_AUTHKEY"] = bridge_key
    model_env["CUDA_VISIBLE_DEVICES"] = model_gpu
    model_env["I4H_RL_SIM_SOCKET"] = str(socket_path)
    print("simulator: " + shlex.join(sim_command), flush=True)
    simulator = subprocess.Popen(sim_command, env=sim_env, cwd=workflows_root)
    try:
        ready_timeout = _sim_ready_timeout()
        deadline = time.monotonic() + ready_timeout
        while not ready_file.is_file():
            return_code = simulator.poll()
            if return_code is not None:
                print(f"Isaac Sim server exited before becoming ready (exit={return_code})", file=sys.stderr)
                return return_code or 1
            if time.monotonic() >= deadline:
                print(f"Isaac Sim server did not become ready within {ready_timeout:g} seconds", file=sys.stderr)
                return 1
            time.sleep(0.25)
        print("launch: " + shlex.join(command), flush=True)
        return_code = subprocess.call(command, env=model_env)
    finally:
        if simulator.poll() is None:
            simulator.terminate()
            try:
                simulator.wait(timeout=15)
            except subprocess.TimeoutExpired:
                simulator.kill()
                simulator.wait()
    if return_code != 0:
        return return_code
    if args.only_eval:
        if not finalize_evaluation(run_dir, metadata, runtime=model_runtime, env=model_env):
            print(
                f"RLinf evaluation exited successfully but produced no complete metrics under {run_dir}",
                file=sys.stderr,
            )
            return 1
        return 0
    if not finalize_training(run_dir, metadata):
        print(f"RLinf training exited successfully but produced no native checkpoint under {run_dir}", file=sys.stderr)
        return 1
    return 0


def export(args: argparse.Namespace, profile: RLProfile, workflows_root: Path) -> int:
    checkpoint_source = resolve_input_path(workflows_root, args.rl_model_path)
    checkpoint_bundle(checkpoint_source, expected_workflow=profile.workflow)
    actor_weights = weights(checkpoint_source)
    resolved_train_config = resolve_input_path(workflows_root, args.train_config) if args.train_config else None
    if resolved_train_config is None:
        resolved_train_config = train_config(checkpoint_source, actor_weights)
    if resolved_train_config is not None and not resolved_train_config.is_file():
        raise SystemExit(f"--train-config does not exist: {resolved_train_config}")
    runtime = _model_runtime_python(workflows_root, args.runtime_python)
    env = _runtime_env(workflows_root, profile)
    _model_preflight(runtime, env, require_two_gpus=False)
    output = resolve_output_path(workflows_root, args.output_dir)
    output.mkdir(parents=True, exist_ok=True)
    command = [
        str(runtime),
        "-m",
        "rlinf.utils.ckpt_convertor.fsdp_convertor.convert_pt_to_hf",
        "--config-name",
        "fsdp_model_convertor",
        f"convertor.ckpt_path={actor_weights}",
        f"convertor.save_path={output}",
        "convertor.torch_dtype=bf16",
    ]
    if resolved_train_config is not None:
        command.append(f"+convertor.train_config_path={resolved_train_config}")
    print(f"workflow: {profile.workflow}")
    print(f"RLinf weights: {actor_weights}")
    print(f"GR00T inference checkpoint: {output}")
    print("launch: " + shlex.join(command), flush=True)
    return_code = subprocess.call(command, env=env)
    expected = (output / "config.json", output / "model.safetensors.index.json")
    if return_code != 0 or not all(path.is_file() and path.stat().st_size > 0 for path in expected):
        print(
            f"RLinf export did not produce a complete GR00T checkpoint (exit={return_code}, output={output})",
            file=sys.stderr,
        )
        return return_code or 1
    manifest = {
        "schema_version": 1,
        "workflow": profile.workflow,
        "scene": profile.scene,
        "trainer": profile.trainer,
        "format": "huggingface",
        "source_checkpoint": str(checkpoint_source),
        "trainer_config": str(resolved_train_config) if resolved_train_config is not None else None,
        "checkpoint": str(output),
        "created_at": datetime.now(UTC).isoformat(),
    }
    write_json(output / "policy.json", manifest)
    print(f"export complete: {output}")
    return 0

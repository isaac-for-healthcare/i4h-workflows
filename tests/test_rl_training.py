# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Contracts for the lightweight Workflow RL training boundary."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest
import yaml

ROOT = Path(__file__).resolve().parent.parent
RL_ROOT = ROOT / "rl"
sys.path.insert(0, str(RL_ROOT))

from i4h_rl.adapters.assemble_trocar import ACTION_KEYS  # noqa: E402
from i4h_rl.adapters.assemble_trocar import _register_gr00t_converters, convert_gr00t_to_workflow_action
from i4h_rl.artifacts import checkpoint_iteration  # noqa: E402
from i4h_rl.artifacts import resolve_input_path, resolve_output_path
from i4h_rl.backends.rlinf import _sim_ready_timeout  # noqa: E402
from i4h_rl.backends.rlinf import (
    checkpoint_bundle,
    checkpoint_root,
    finalize_evaluation,
    finalize_training,
    train_config,
    weights,
)
from i4h_rl.backends.rsl_rl import _runtime_env as rsl_rl_runtime_env
from i4h_rl.contract import validate_workflow_contract  # noqa: E402
from i4h_rl.profile import RLProfile  # noqa: E402
from i4h_rl.profile import ProfileError, available_profiles, load_profile
from i4h_rl.sim_bridge import to_numpy_tree  # noqa: E402
from i4h_rl.sim_bridge import RemoteIsaacEnv, to_torch_tree


def test_trocar_profile_matches_scene_and_rlinf_config() -> None:
    profile = load_profile("assemble_trocar")
    scene = yaml.safe_load((ROOT / "arena/i4h_arena/scenes/manifest/g1_trocar.yaml").read_text())
    task = yaml.safe_load((ROOT / "tasks/gr00t_n15/i4h_tasks/gr00t_n15/manifest/assemble_trocar.yaml").read_text())
    config = yaml.safe_load(profile.trainer_config.read_text())

    assert profile.scene == "g1_trocar"
    assert profile.action_dof == scene["dof"]
    assert profile.cameras == tuple(scene["cameras"])
    assert config["env"]["train"]["max_episode_steps"] == scene["max_steps"]
    assert config["env"]["train"]["init_params"]["id"] == profile.train_task_id
    assert config["env"]["eval"]["init_params"]["id"] == profile.eval_task_id
    assert config["env"]["train"]["init_params"]["task_description"] == task["summary"].lower()
    assert config["actor"]["model"]["action_dim"] == profile.policy_action_dof
    assert config["weight_syncer"]["type"] == "patch"
    assert config["actor"]["fsdp_config"]["amp_autocast"]["enabled"] is False
    assert config["actor"]["fsdp_config"]["grad_scaler"]["enabled"] is False
    assert config["env"]["train"]["isaaclab"]["action_mapping"]["prefix_pad"] == (
        profile.action_dof - profile.policy_action_dof
    )


def test_gr00t_action_mapping_is_ordered_and_zero_pads_uncontrolled_body() -> None:
    chunks = {key: np.full((2, 3, 7), index + 1, dtype=np.float32) for index, key in enumerate(reversed(ACTION_KEYS))}
    # Rebuild in deliberately reversed insertion order. The converter must use
    # semantic modality order, not dict insertion order.
    chunks = {key: chunks[key] for key in reversed(ACTION_KEYS)}
    result = convert_gr00t_to_workflow_action(chunks, chunk_size=2)

    assert result.shape == (2, 2, 43)
    np.testing.assert_array_equal(result[..., :15], 0.0)
    for index, key in enumerate(ACTION_KEYS):
        expected = chunks[key][..., :2, :]
        np.testing.assert_array_equal(result[..., 15 + index * 7 : 22 + index * 7], expected)


def test_workflow_observation_mapping_uses_current_arena_term_names() -> None:
    torch = pytest.importorskip("torch")
    from i4h_rl.adapters.assemble_trocar import convert_workflow_obs_to_gr00t, wrap_workflow_observation

    batch, height, width = 2, 4, 5
    policy = {
        "front_camera_rgb": torch.zeros((batch, height, width, 4), dtype=torch.uint8),
        "left_wrist_camera_rgb": torch.ones((batch, height, width, 4), dtype=torch.uint8),
        "right_wrist_camera_rgb": torch.full((batch, height, width, 4), 2, dtype=torch.uint8),
        # Current Arena publishes 29 positions, velocities, and applied
        # torques. GR00T consumes the 14 controlled arm positions below.
        "robot_joint_state": torch.arange(batch * 87, dtype=torch.float32).reshape(batch, 87),
        "robot_dex3_joint_state": torch.arange(batch * 14, dtype=torch.float32).reshape(batch, 14),
    }
    bridge = wrap_workflow_observation({"policy": policy}, task_description="install trocar from box", num_envs=batch)
    gr00t = convert_workflow_obs_to_gr00t(bridge)

    assert bridge["main_images"].shape == (batch, height, width, 3)
    assert bridge["extra_view_images"].shape == (batch, 2, height, width, 3)
    assert bridge["states"].shape == (batch, 28)
    torch.testing.assert_close(bridge["states"][:, :14], policy["robot_joint_state"][:, 15:29])
    assert gr00t["video.room_view"].shape == (batch, 1, height, width, 3)
    assert gr00t["state.left_arm"].shape == (batch, 1, 7)
    assert gr00t["state.right_hand"].shape == (batch, 1, 7)
    assert gr00t["annotation.human.task_description"] == ["install trocar from box"] * batch
    assert "annotation.human.action.task_description" not in gr00t


def test_sim_bridge_round_trips_nested_tensor_payloads() -> None:
    torch = pytest.importorskip("torch")
    payload = {
        "policy": {
            "image": torch.arange(24, dtype=torch.uint8).reshape(1, 2, 3, 4),
            "state": torch.tensor([[1.0, 2.0]], dtype=torch.float32),
        },
        "done": torch.tensor([False]),
    }

    transported = to_numpy_tree(payload)
    restored = to_torch_tree(transported, device="cpu")

    torch.testing.assert_close(restored["policy"]["image"], payload["policy"]["image"])
    torch.testing.assert_close(restored["policy"]["state"], payload["policy"]["state"])
    torch.testing.assert_close(restored["done"], payload["done"])


def test_sim_bridge_close_is_idempotent_when_server_has_exited() -> None:
    class ClosedConnection:
        def __init__(self) -> None:
            self.send_count = 0
            self.close_count = 0

        def send(self, _payload) -> None:
            self.send_count += 1
            raise BrokenPipeError

        def close(self) -> None:
            self.close_count += 1

    connection = ClosedConnection()
    bridge = RemoteIsaacEnv.__new__(RemoteIsaacEnv)
    bridge.socket_path = "/tmp/not-connected.sock"
    bridge._device = "cpu"
    bridge._connection = connection
    bridge._closed = False

    bridge.close()
    bridge.close()

    assert connection.send_count == 1
    assert connection.close_count == 1


def test_train_rl_list_and_dry_run_do_not_require_simulator(tmp_path: Path) -> None:
    env = os.environ.copy()
    env["I4H_WORKFLOWS"] = str(ROOT)
    run_parent = ROOT / "runs/assemble_trocar"
    before = set(run_parent.iterdir()) if run_parent.exists() else set()
    listed = subprocess.run(
        [str(ROOT / "train.sh"), "rl", "list"],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "assemble_trocar\trlinf\tppo_actor_critic\tg1_trocar" in listed.stdout

    model = tmp_path / "sft-checkpoint"
    model.mkdir()
    dry = subprocess.run(
        [
            str(ROOT / "train.sh"),
            "rl",
            "assemble_trocar",
            "--model-path",
            str(model),
            "--num-envs",
            "2",
            "--epochs",
            "1",
            "--dry-run",
        ],
        cwd=ROOT,
        env=env,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "actions: 28-D policy -> 43-D scene" in dry.stdout
    assert "parallel envs: 2" in dry.stdout
    assert "max epochs: 1" in dry.stdout
    assert "simulator and trainer were not launched" in dry.stdout
    after = set(run_parent.iterdir()) if run_parent.exists() else set()
    assert after == before


def test_train_rl_show_validates_the_full_profile_contract() -> None:
    shown = subprocess.run(
        [str(ROOT / "train.sh"), "rl", "show", "ultrasound_probe_reach"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "workflow: ultrasound_probe_reach" in shown.stdout
    assert "trainer: rsl_rl (ppo)" in shown.stdout


def test_supported_workflows_have_rl_profiles() -> None:
    assert set(available_profiles()) == {"assemble_trocar", "ultrasound_probe_reach"}


def test_profiles_declare_workflow_adapters_and_simulator_contracts() -> None:
    trocar = load_profile("assemble_trocar")
    ultrasound = load_profile("ultrasound_probe_reach")

    assert trocar.adapter_module == "i4h_rl.adapters.assemble_trocar"
    assert trocar.simulation.enable_cameras is True
    assert trocar.simulation.env_spacing == 6.0
    assert ultrasound.adapter_module == "i4h_rl.adapters.ultrasound_probe_reach"
    assert ultrasound.simulation.enable_cameras is False
    assert ultrasound.simulation.env_spacing == 2.0


def test_supported_profiles_match_workflow_and_backend_contracts() -> None:
    for profile_path in available_profiles().values():
        profile = RLProfile.load(profile_path)
        validate_workflow_contract(profile, ROOT)
        backend = __import__(f"i4h_rl.backends.{profile.trainer}", fromlist=["validate_profile"])
        backend.validate_profile(profile, ROOT)


def test_profile_schema_accepts_a_new_rsl_workflow_without_core_registration(tmp_path: Path) -> None:
    config = tmp_path / "example_agent.yaml"
    config.touch()
    profile_path = tmp_path / "example_reach.yaml"
    profile_path.write_text(
        """schema_version: 1
workflow: example_reach
scene: example_reach
trainer: rsl_rl
algorithm: ppo
adapter_module: example_package.example_reach
trainer_config: example_agent.yaml
train_task_id: example_reach
eval_task_id: example_reach
task_description: reach the target
action_dof: 6
policy_action_dof: 6
state_dof: 20
cameras: []
default_num_envs: 32
default_epochs: 100
simulation:
  env_spacing: 2.5
  presets: physx
  enable_cameras: false
""",
        encoding="utf-8",
    )

    profile = RLProfile.load(profile_path)

    assert profile.workflow == "example_reach"
    assert profile.adapter_module == "example_package.example_reach"
    assert profile.simulation.env_spacing == 2.5


def test_profile_schema_rejects_unknown_fields(tmp_path: Path) -> None:
    config = tmp_path / "example_agent.yaml"
    config.touch()
    profile_path = tmp_path / "example_reach.yaml"
    profile_path.write_text(
        """schema_version: 1
workflow: example_reach
scene: example_reach
trainer: rsl_rl
algorithm: ppo
trainer_config: example_agent.yaml
train_task_id: example_reach
eval_task_id: example_reach
task_description: reach the target
action_dof: 6
policy_action_dof: 6
state_dof: 20
cameras: []
default_num_envs: 32
default_epochs: 100
simulation:
  env_spacing: 2.5
  presets: physx
  enable_cameras: false
typo_field: true
""",
        encoding="utf-8",
    )

    with pytest.raises(ProfileError, match="unknown fields: typo_field"):
        RLProfile.load(profile_path)


def test_generic_rlinf_extension_delegates_to_selected_adapter(monkeypatch) -> None:
    from i4h_rl import extension

    module_name = "test_selected_rl_adapter"
    adapter = ModuleType(module_name)
    calls: list[str] = []
    adapter.register = lambda: calls.append("register")
    monkeypatch.setitem(sys.modules, module_name, adapter)
    monkeypatch.setenv("I4H_RL_ADAPTER_MODULE", module_name)
    extension._registered.discard(module_name)

    extension.register()
    extension.register()

    assert calls == ["register"]


def test_sim_server_accepts_profile_selected_scene_settings() -> None:
    from i4h_rl.sim_server import _parser, _scene_args

    args = _parser().parse_args(
        [
            "--scene",
            "future_scene",
            "--socket",
            "/tmp/future.sock",
            "--ready-file",
            "/tmp/future.ready",
            "--num-envs",
            "8",
            "--max-episode-steps",
            "200",
            "--env-spacing",
            "3.5",
            "--presets",
            "physx",
            "--no-enable-cameras",
        ]
    )
    scene_args = _scene_args(args)

    assert args.scene == "future_scene"
    assert scene_args.num_envs == 8
    assert scene_args.episode_steps == 200
    assert scene_args.env_spacing == 3.5
    assert scene_args.no_cameras is True


def test_probe_reach_profile_uses_rsl_rl_without_images() -> None:
    profile = load_profile("ultrasound_probe_reach")
    assert profile.trainer == "rsl_rl"
    assert profile.algorithm == "ppo"
    assert profile.scene == "ultrasound_probe_reach"
    assert profile.action_dof == profile.policy_action_dof == 6
    assert profile.state_dof == 34
    assert profile.cameras == ()
    assert profile.trainer_config.parent.name == "config"
    assert profile.trainer_config.name == "ultrasound_probe_reach_ppo_rsl_rl.yaml"


def test_rsl_rl_runtime_accepts_the_kit_eula_noninteractively(monkeypatch) -> None:
    monkeypatch.delenv("OMNI_KIT_ACCEPT_EULA", raising=False)

    env = rsl_rl_runtime_env(ROOT, load_profile("ultrasound_probe_reach"))

    assert env["OMNI_KIT_ACCEPT_EULA"] == "YES"


def test_probe_reach_dry_run_needs_no_starting_model() -> None:
    dry = subprocess.run(
        [
            str(ROOT / "train.sh"),
            "rl",
            "ultrasound_probe_reach",
            "--num-envs",
            "4",
            "--epochs",
            "2",
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert "trainer: rsl_rl (ppo)" in dry.stdout
    assert "observations: 34-D state" in dry.stdout
    assert "actions: 6-D policy -> 6-D scene" in dry.stdout


def test_probe_reach_rejects_a_starting_model_even_for_dry_run(tmp_path: Path) -> None:
    model = tmp_path / "not-used-by-rsl-rl"
    model.touch()

    result = subprocess.run(
        [
            str(ROOT / "train.sh"),
            "rl",
            "ultrasound_probe_reach",
            "--model-path",
            str(model),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "trains from scratch" in result.stderr


def test_rl_cli_rejects_an_extra_positional_argument() -> None:
    result = subprocess.run(
        [str(ROOT / "train.sh"), "rl", "ultrasound_probe_reach", "unexpected"],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "unexpected positional argument" in result.stderr


def test_rl_export_rejects_silently_ignored_operation_flags(tmp_path: Path) -> None:
    checkpoint = tmp_path / "model.pt"
    checkpoint.touch()
    result = subprocess.run(
        [
            str(ROOT / "train.sh"),
            "rl",
            "export",
            "ultrasound_probe_reach",
            "--rl-model-path",
            str(checkpoint),
            "--output-dir",
            str(tmp_path / "exported"),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=False,
    )

    assert result.returncode != 0
    assert "export does not accept --dry-run" in result.stderr


def test_rlinf_actor_checkpoint_is_resolved_for_export(tmp_path: Path) -> None:
    actor_weights = tmp_path / "checkpoint/actor/model_state_dict/full_weights.pt"
    actor_weights.parent.mkdir(parents=True)
    actor_weights.touch()

    assert weights(tmp_path / "checkpoint") == actor_weights.resolve()
    assert weights(actor_weights) == actor_weights.resolve()
    assert checkpoint_root(actor_weights) == (tmp_path / "checkpoint").resolve()


def test_rlinf_export_discovers_resolved_run_config(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    actor_weights = run_dir / "checkpoints/global_step_4/actor/model_state_dict/full_weights.pt"
    actor_weights.parent.mkdir(parents=True)
    actor_weights.touch()
    config = run_dir / "tensorboard/config.yaml"
    config.parent.mkdir()
    config.write_text("actor:\n  model:\n    model_type: gr00t\n", encoding="utf-8")

    assert train_config(run_dir, actor_weights) == config.resolve()


def test_rlinf_training_finalizes_a_portable_checkpoint_bundle(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    actor_weights = run_dir / "checkpoints/global_step_4/actor/model_state_dict/full_weights.pt"
    actor_weights.parent.mkdir(parents=True)
    actor_weights.touch()
    metadata: dict[str, object] = {
        "workflow": "assemble_trocar",
        "model_path": "/models/groot-sft",
        "trainer_config": "/repo/rl/config/assemble_trocar_ppo_gr00t.yaml",
    }

    assert finalize_training(run_dir, metadata)
    bundle = yaml.safe_load((run_dir / "checkpoint.json").read_text(encoding="utf-8"))
    assert bundle["format"] == "rlinf-fsdp"
    assert bundle["weights"] == "checkpoints/global_step_4/actor/model_state_dict/full_weights.pt"
    assert weights(run_dir) == actor_weights.resolve()


def test_rlinf_checkpoint_bundle_rejects_another_workflow(tmp_path: Path) -> None:
    run_dir = tmp_path / "run"
    run_dir.mkdir()
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "rlinf-fsdp",
                "workflow": "another_workflow",
                "checkpoint": "checkpoints/global_step_4",
                "weights": "checkpoints/global_step_4/actor/model_state_dict/full_weights.pt",
                "base_model": "/models/groot-sft",
                "trainer_config": "/repo/rl/config/example.yaml",
            }
        ),
        encoding="utf-8",
    )

    with pytest.raises(SystemExit, match="does not match selected workflow"):
        checkpoint_bundle(run_dir, expected_workflow="assemble_trocar")


@pytest.mark.parametrize("value", ["not-a-number", "0", "-1", "nan", "inf"])
def test_rlinf_simulator_ready_timeout_must_be_positive_and_finite(monkeypatch, value: str) -> None:
    monkeypatch.setenv("I4H_RL_SIM_READY_TIMEOUT_S", value)

    with pytest.raises(SystemExit, match="must be a positive number"):
        _sim_ready_timeout()


def test_rlinf_evaluation_requires_and_records_tensorboard_metrics(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "eval"
    event = run_dir / "tensorboard/events.out.tfevents.test"
    event.parent.mkdir(parents=True)
    event.write_bytes(b"event")
    metrics = {
        "eval/return": 1.25,
        "eval/success_once": 1.0,
        "eval/episode_len": 42.0,
        "eval/num_trajectories": 2.0,
    }

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(metrics), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    metadata = {
        "workflow": "assemble_trocar",
        "scene": "g1_trocar",
        "trainer": "rlinf",
        "native_checkpoint": "/tmp/native-checkpoint",
    }

    assert finalize_evaluation(run_dir, metadata, runtime=Path("/python"), env={})
    evaluation = json.loads((run_dir / "evaluation.json").read_text())
    assert evaluation["checkpoint"] == "/tmp/native-checkpoint"
    assert evaluation["metrics"] == metrics
    assert json.loads((run_dir / "run.json").read_text())["metrics"] == metrics


def test_rlinf_evaluation_records_metrics_but_fails_without_task_success(tmp_path: Path, monkeypatch) -> None:
    run_dir = tmp_path / "eval"
    event = run_dir / "tensorboard/events.out.tfevents.test"
    event.parent.mkdir(parents=True)
    event.write_bytes(b"event")
    metrics = {
        "eval/return": 1.25,
        "eval/success_once": 0.0,
        "eval/episode_len": 42.0,
        "eval/num_trajectories": 2.0,
    }

    def fake_run(*_args, **_kwargs):
        return SimpleNamespace(returncode=0, stdout=json.dumps(metrics), stderr="")

    monkeypatch.setattr(subprocess, "run", fake_run)
    metadata = {
        "workflow": "assemble_trocar",
        "scene": "g1_trocar",
        "trainer": "rlinf",
        "native_checkpoint": "/tmp/native-checkpoint",
    }

    assert not finalize_evaluation(run_dir, metadata, runtime=Path("/python"), env={})
    assert json.loads((run_dir / "evaluation.json").read_text())["metrics"] == metrics


def test_trocar_eval_resolves_base_model_from_checkpoint_bundle(tmp_path: Path) -> None:
    base_model = tmp_path / "groot-sft"
    base_model.mkdir()
    run_dir = tmp_path / "train-run"
    weights = run_dir / "checkpoints/global_step_4/actor/model_state_dict/full_weights.pt"
    weights.parent.mkdir(parents=True)
    weights.touch()
    (run_dir / "checkpoint.json").write_text(
        json.dumps(
            {
                "schema_version": 1,
                "format": "rlinf-fsdp",
                "workflow": "assemble_trocar",
                "checkpoint": "checkpoints/global_step_4",
                "base_model": str(base_model),
                "weights": "checkpoints/global_step_4/actor/model_state_dict/full_weights.pt",
                "trainer_config": "/repo/rl/config/assemble_trocar_ppo_gr00t.yaml",
            }
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            str(ROOT / "train.sh"),
            "rl",
            "assemble_trocar",
            "--eval",
            "--checkpoint",
            str(run_dir),
            "--dry-run",
        ],
        cwd=ROOT,
        text=True,
        capture_output=True,
        check=True,
    )

    assert f"model: {base_model}" in result.stdout
    assert "operation: evaluation" in result.stdout


def test_rl_cli_paths_accept_workflow_and_repo_relative_forms(tmp_path: Path) -> None:
    checkpoint = ROOT / "runs/path-contract/model.pt"
    checkpoint.parent.mkdir(parents=True, exist_ok=True)
    checkpoint.touch()
    try:
        assert resolve_output_path(ROOT, "runs/probe") == ROOT / "runs/probe"
        assert resolve_output_path(ROOT, "./runs/probe") == ROOT / "runs/probe"
        assert resolve_input_path(ROOT, "runs/path-contract/model.pt") == checkpoint
        assert resolve_input_path(ROOT, "./runs/path-contract/model.pt") == checkpoint
    finally:
        checkpoint.unlink()
        checkpoint.parent.rmdir()


def test_rsl_checkpoints_sort_by_iteration_not_filename() -> None:
    checkpoints = [Path("model_75.pt"), Path("model_199.pt"), Path("model_100.pt")]
    assert sorted(checkpoints, key=checkpoint_iteration)[-1] == Path("model_199.pt")


def test_trocar_registers_the_n15_action_converter() -> None:
    simulation_io = SimpleNamespace(OBS_CONVERSION={}, ACTION_CONVERSION_N1D5={})

    _register_gr00t_converters(simulation_io)

    assert simulation_io.OBS_CONVERSION["i4h_g1_dex3"] is not None
    assert simulation_io.ACTION_CONVERSION_N1D5["i4h_g1_dex3"] is convert_gr00t_to_workflow_action

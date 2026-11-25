# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Exercise the patched RLinf factory and optimizer loop without policy downloads."""

import ast
import logging
import re
import runpy
import shutil
import subprocess
import sys
import time
from pathlib import Path
from types import ModuleType, SimpleNamespace
from unittest.mock import Mock

import numpy as np
import pytest
from omegaconf import DictConfig, OmegaConf

torch = pytest.importorskip("torch")
ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / "third_party/RLinf-a4b6abe/rlinf"


def _function(relative_path, name, namespace, owner=None):
    path = SOURCE / relative_path
    if not path.is_file():
        pytest.skip(f"optional patched RLinf source is absent: {path}")
    tree = ast.parse(path.read_text())
    if owner:
        tree = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == owner)
    node = next(node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name == name)
    node.decorator_list = []
    exec(compile(ast.Module(body=[node], type_ignores=[]), str(path), "exec"), namespace)
    return namespace[name]


@pytest.mark.parametrize("available", [True, False])
def test_model_factory_installs_patch_before_loading_checkpoint(monkeypatch, tmp_path, available):
    # Execute the entire vendor factory, replacing only optional model libraries.
    # The numeric equivalence tests live with the real GR00T N1.7 task runtime.
    install = Mock(return_value=available)
    model = torch.nn.Linear(2, 2)

    def from_pretrained(**kwargs):
        install.assert_called_once_with()
        assert kwargs["local_model_path"] == str(tmp_path)
        return model

    def module(name, **attributes):
        value = ModuleType(name)
        value.__dict__.update(attributes)
        monkeypatch.setitem(sys.modules, name, value)

    module("i4h_tasks.gr00t_n17._patch_embed", install_linear_patch_embed=install)
    module("gr00t.configs.model.gr00t_n1d7", Gr00tN1d7Config=Mock())
    module("gr00t.model.gr00t_n1d7.gr00t_n1d7", Gr00tN1d7=Mock())
    module("transformers", AutoConfig=Mock(), AutoModel=Mock())
    module("rlinf.utils.patcher", Patcher=Mock())
    module("gr00t.data.embodiment_tags", EmbodimentTag=Mock())
    module(
        "rlinf.models.embodiment.gr00t.gr00t_n1d7.gr00t_action_model",
        GR00T_N1_7_ForRLActionPrediction=SimpleNamespace(from_pretrained=from_pretrained),
    )
    module("rlinf.models.embodiment.gr00t.utils", replace_dropout_with_identity=Mock())
    factory = _function(
        "models/embodiment/gr00t/gr00t_n1d7/__init__.py",
        "get_model",
        dict(torch=torch, DictConfig=DictConfig, OmegaConf=OmegaConf, Path=Path, get_logger=logging.getLogger),
    )
    cfg = OmegaConf.create(
        dict(
            model_path=str(tmp_path),
            embodiment_tag="new_embodiment",
            denoising_steps=4,
            num_action_chunks=1,
            obs_converter_type="workflow_catheter",
            rl_head_config=dict(add_value_head=False, disable_dropout=False),
        )
    )
    if available:
        assert factory(cfg) is model
        assert model.weight.dtype == torch.bfloat16
    else:
        with pytest.raises(RuntimeError, match="linear patch embedding"):
            factory(cfg)
        install.assert_called_once_with()


def _actor(monkeypatch, rank=0, world_size=1, fail_step=None):
    run_training = _function(
        "workers/actor/fsdp_actor_worker.py",
        "run_training",
        {},
        owner="EmbodiedFSDPActor",
    )
    nested = runpy.run_path(str(SOURCE / "utils/nested_dict_process.py"))
    run_training.__globals__.update(
        nested,
        time=time,
        clear_memory=Mock(),
        all_reduce_dict=lambda data, **kwargs: data,
    )
    _function("workers/actor/fsdp_actor_worker.py", "process_nested_dict_for_train", run_training.__globals__)
    _function("utils/metric_utils.py", "append_to_dict", run_training.__globals__)
    monkeypatch.setattr(torch.distributed, "get_world_size", lambda: world_size)
    model = torch.nn.Linear(1, 1)
    optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
    actor = SimpleNamespace(
        _rank=rank,
        _world_size=world_size,
        is_weight_offloaded=False,
        is_optimizer_offloaded=False,
        model=model,
        optimizer=optimizer,
        optimizer_steps=10,  # Resumed training keeps the global count.
        gradient_accumulation=2,
        lr_scheduler=Mock(),
        torch_platform=SimpleNamespace(empty_cache=Mock()),
        log_info=Mock(),
        cfg=OmegaConf.create(
            dict(
                actor=dict(seed=0, global_batch_size=4 * world_size, micro_batch_size=2), algorithm=dict(update_epoch=2)
            )
        ),
        rollout_batch={"prev_logprobs": torch.ones(4, 2, 1)},
    )
    micro_batches = []

    def train_micro_batch(*, micro_batch, metrics, is_last):
        actor.model(micro_batch["prev_logprobs"]).square().mean().backward()
        micro_batches.append(is_last)

    def optimizer_step():
        if actor.optimizer_steps + 1 == fail_step:
            raise RuntimeError("injected optimizer failure")
        actor.optimizer.step()
        actor.optimizer_steps += 1
        return 1.0, [0.01]

    actor.train_micro_batch = train_micro_batch
    actor.optimizer_step = optimizer_step
    return actor, run_training, micro_batches


def test_optimizer_progress_tracks_steps_across_ppo_epochs(monkeypatch):
    actor, run_training, micro_batches = _actor(monkeypatch)
    before = actor.model.weight.detach().clone()
    metrics = run_training(actor)

    messages = [call.args[0] for call in actor.log_info.call_args_list]
    assert messages[0] == "PPO optimization started: 4 optimizer steps, 2 micro-batches per step"
    assert len(messages) == 5
    for step, message in enumerate(messages[1:], 1):
        assert f"PPO optimizer step {10 + step} finished ({step}/4 in this update, " in message
        assert message.endswith("s)")
    assert micro_batches == [False, True] * 4
    assert not torch.equal(before, actor.model.weight)
    assert all(np.isfinite(value) for value in metrics.values())
    actor.lr_scheduler.step.assert_called_once_with()


def test_optimizer_progress_only_logs_rank_zero(monkeypatch):
    actor, run_training, micro_batches = _actor(monkeypatch, rank=1, world_size=2)
    run_training(actor)
    actor.log_info.assert_not_called()
    assert len(micro_batches) == 8


def test_failed_optimizer_step_is_not_reported_finished(monkeypatch):
    actor, run_training, _ = _actor(monkeypatch, fail_step=12)
    with pytest.raises(RuntimeError, match="injected optimizer failure"):
        run_training(actor)
    messages = [call.args[0] for call in actor.log_info.call_args_list]
    assert len(messages) == 2
    assert "step 11 finished (1/4" in messages[-1]
    actor.lr_scheduler.step.assert_not_called()


# 0 is a fresh checkout, 1 an rc3 checkout and 2 an rc4 checkout that already carry the older patches.
@pytest.mark.parametrize("existing_patches", [0, 1, 2])
@pytest.mark.parametrize("installer", ["cpu", "full"])
def test_setup_applies_worker_patch_and_can_be_repeated(tmp_path, existing_patches, installer):
    if not SOURCE.is_dir():
        pytest.skip(f"optional RLinf source is absent: {SOURCE}")
    checkout = tmp_path / "third_party/RLinf-a4b6abe"
    checkout.parent.mkdir()

    def run(*args):
        return subprocess.run(args, check=True, capture_output=True, text=True)

    run("git", "clone", "--shared", "--no-checkout", str(SOURCE.parent), str(checkout))
    run("git", "-C", str(checkout), "checkout", "--detach", "a4b6abe205d7942f45cf3e8843c3e72ce818729d")
    patches = ["rlinf_training_failures.patch", "rlinf_gr00t_training.patch", "rlinf_channel_failures.patch"]
    for name in patches:
        shutil.copyfile(ROOT / "third_party" / name, checkout.parent / name)
    for name in patches[:existing_patches]:
        run("git", "-C", str(checkout), "apply", str(checkout.parent / name))
    sentinel = checkout / "local-notes.txt"
    sentinel.write_text("preserve local work\n")

    if installer == "cpu":
        script = tmp_path / "scripts/setup_rl_cpu.sh"
        script.parent.mkdir()
        shutil.copyfile(ROOT / "scripts/setup_rl_cpu.sh", script)
        command = ("bash", str(script))
    else:
        # Run the production patch helper and its RLinf calls, without fetching
        # all the unrelated simulator/model dependencies from setup.sh.
        source = (ROOT / "third_party/setup.sh").read_text()
        helper = re.search(r"^apply_patch_once\(\) \{\n.*?^\}\n", source, re.M | re.S).group()
        block = re.search(
            r'^if \[\[ -d "\$\{THIRD_PARTY_DIR\}/\$\{RLINF_DIR\}/\.git" \]\]; then\n.*?^fi', source, re.M | re.S
        ).group()
        script = f'set -euo pipefail\nLOG_PREFIX=test\nTHIRD_PARTY_DIR="$1"\nRLINF_DIR=RLinf-a4b6abe\n{helper}{block}\n'
        command = ("bash", "-c", script, "setup", str(checkout.parent))
    run(*command)
    run(*command)
    for name in patches:
        run("git", "-C", str(checkout), "apply", "--reverse", "--check", str(checkout.parent / name))
    assert sentinel.read_text() == "preserve local work\n"

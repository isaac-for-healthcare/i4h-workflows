# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PI0 is compiled before a session is ready, and on a GPU the pinned XLA can build (NVBug 6935977)."""

from __future__ import annotations

import numpy as np
import pytest

from i4h_common.server import Session
from i4h_tasks.openpi_pi0 import server


def _session() -> Session:
    return Session(
        task_uid="uid",
        task_id="openpi_pi0/ultrasound_liver_scan",
        run_id="run",
        episode_index=0,
        prompt="Perform a liver ultrasound.",
        checkpoint="",
    )


@pytest.mark.parametrize("capability", [(10, 3), (12, 1)])
def test_triton_gemm_is_disabled_on_a_gpu_the_pinned_xla_does_not_know(monkeypatch, capability) -> None:
    """GB300 (10.3) and GB10 (12.1) abort in Triton GEMM compilation under jaxlib 0.5.3."""
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/dump")

    server.configure_xla(capability)

    assert server.os.environ["XLA_FLAGS"] == "--xla_dump_to=/tmp/dump --xla_gpu_enable_triton_gemm=false"


@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0), (10, 0), (10, 1), (12, 0), None])
def test_known_gpus_keep_the_faster_triton_gemms(monkeypatch, capability) -> None:
    monkeypatch.delenv("XLA_FLAGS", raising=False)

    server.configure_xla(capability)

    assert "XLA_FLAGS" not in server.os.environ


def test_an_explicit_triton_gemm_setting_wins(monkeypatch) -> None:
    monkeypatch.setenv("XLA_FLAGS", "--xla_gpu_enable_triton_gemm=true")

    server.configure_xla((10, 3))

    assert server.os.environ["XLA_FLAGS"] == "--xla_gpu_enable_triton_gemm=true"


def test_the_warm_up_runs_the_same_observation_path_as_a_real_frame(monkeypatch) -> None:
    """A different shape or key set would compile a second program on the first real frame."""
    pytest.importorskip("openpi_client")
    seen: list[dict] = []
    policy = type("Policy", (), {"infer": lambda self, observation: seen.append(observation) or {}})()

    server.OpenPiPi0Server._warm_up(object.__new__(server.OpenPiPi0Server), policy, _session())

    (observation,) = seen
    assert set(observation) == {"observation/state", "observation/image", "observation/wrist_image", "prompt"}
    assert observation["observation/state"].shape == (server.STATE_DIM,)
    assert observation["observation/image"].shape == (*server.IMAGE_SIZE, 3)
    assert observation["observation/wrist_image"].dtype == np.uint8
    assert observation["prompt"] == "Perform a liver ultrasound."


@pytest.mark.parametrize("ids", [[], ["local/one", "local/two"]])
def test_checkpoint_missing_or_ambiguous_stats_fail_before_openpi_import(tmp_path, ids):
    for identity in ids:
        stats = tmp_path / "assets" / identity / "norm_stats.json"
        stats.parent.mkdir(parents=True)
        stats.write_text("{}")
    session = _session()
    session.checkpoint = str(tmp_path)
    instance = object.__new__(server.OpenPiPi0Server)
    instance._policies = {}
    with pytest.raises(ValueError, match="expected exactly one normalization asset"):
        instance.load(session)


def test_checkpoint_identity_uses_saved_asset_without_training_dataset(tmp_path):
    stats = tmp_path / "assets" / "custom" / "ultrasound_e2e" / "norm_stats.json"
    stats.parent.mkdir(parents=True)
    stats.write_text("{}")
    session = _session()
    session.checkpoint = str(tmp_path)
    assert server.checkpoint_asset_id(session.checkpoint) == "custom/ultrasound_e2e"
    instance = object.__new__(server.OpenPiPi0Server)
    assert instance._cache_key(session).endswith("|custom/ultrasound_e2e")


def test_pretrained_cache_separates_manifest_normalization_ids():
    first, second = _session(), _session()
    first.model = {"repo": "model", "repo_id": "one/data"}
    second.model = {"repo": "model", "repo_id": "two/data"}
    instance = object.__new__(server.OpenPiPi0Server)
    assert instance._cache_key(first) != instance._cache_key(second)


@pytest.mark.parametrize("entry", ["train", "server"])
def test_xla_is_configured_before_the_first_openpi_import(monkeypatch, tmp_path, entry):
    import builtins

    from i4h_tasks.openpi_pi0 import train

    module = train if entry == "train" else server
    monkeypatch.setattr(module, "cuda_compute_capability", lambda: (10, 3))
    monkeypatch.setenv("XLA_FLAGS", "--xla_dump_to=/tmp/rc5")
    original = builtins.__import__

    class ReachedOpenPi(Exception):
        pass

    def guarded(name, *args, **kwargs):
        if name == "openpi" or name.startswith("openpi."):
            assert server.os.environ["XLA_FLAGS"] == ("--xla_dump_to=/tmp/rc5 --xla_gpu_enable_triton_gemm=false")
            raise ReachedOpenPi
        return original(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", guarded)
    with pytest.raises(ReachedOpenPi):
        if entry == "train":
            train.main(["--task", "openpi_pi0/ultrasound_liver_scan", "--dataset", str(tmp_path)])
        else:
            instance = object.__new__(server.OpenPiPi0Server)
            instance._policies = {}
            instance.load(_session())


@pytest.mark.parametrize("override", [False, True])
def test_loading_resolves_real_openpi_normalization_without_dataset(monkeypatch, tmp_path, override):
    pytest.importorskip("openpi")
    from openpi.policies import policy_config
    from openpi.shared import normalize

    identity = "custom/ultrasound_e2e" if override else "manifest/pretrained"
    mean = np.arange(7, dtype=np.float32)
    normalize.save(
        tmp_path / "assets" / identity,
        {
            "state": normalize.NormStats(mean=mean, std=np.ones(7)),
            "actions": normalize.NormStats(mean=np.zeros(6), std=np.ones(6)),
        },
    )
    session = _session()
    session.model = {"repo": str(tmp_path), "repo_id": "manifest/pretrained"}
    session.checkpoint = str(tmp_path) if override else ""
    monkeypatch.setenv("LEROBOT_HOME", str(tmp_path / "absent-dataset"))
    seen = []

    def create(config, path):
        # This is the pinned policy loader's real data factory and asset lookup.
        data = config.data.create(config.assets_dirs, config.model)
        assert data.asset_id == identity
        stats = normalize.load(tmp_path / "assets" / data.asset_id)
        np.testing.assert_array_equal(stats["state"].mean, mean)
        seen.append((config, path))
        return object()

    monkeypatch.setattr(policy_config, "create_trained_policy", create)
    monkeypatch.setattr(server.OpenPiPi0Server, "_warm_up", lambda *_: None)
    instance = object.__new__(server.OpenPiPi0Server)
    instance._policies = {}
    instance.load(session)
    instance.load(session)
    assert len(seen) == 1
    assert len(instance._policies) == 1

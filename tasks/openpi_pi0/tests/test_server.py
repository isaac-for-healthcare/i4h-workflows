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


@pytest.mark.parametrize("capability", [(8, 6), (8, 9), (9, 0), (10, 0), (12, 0), None])
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

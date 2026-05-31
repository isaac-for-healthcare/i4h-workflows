#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Build TRT engines for gr00t_n17 (GR00T N1.7) without a LeRobot dataset.

Drives the upstream ``build_trt_pipeline.py`` end-to-end (export ONNX →
build TRT engines → verify → benchmark), but replaces the LeRobot dataset
read with a synthetic observation captured from the model's own modality
config. No recorded trajectory needed.

Usage:
    build_trt_engines.py --env scissor_pick_and_place
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

import numpy as np
from common.config import get_environment_config
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
AGENTIC = REPO.parents[1]  # workflows/agentic
GR00T = AGENTIC / "third_party" / "Isaac-GR00T-1.7"
SUPPORTED = ("scissor_pick_and_place",)

for m in [k for k in sys.modules if k == "gr00t" or k.startswith("gr00t.")]:
    del sys.modules[m]
sys.path.insert(0, str(GR00T))
sys.path.insert(0, str(GR00T / "scripts" / "deployment"))

import build_trt_pipeline as build  # noqa: E402
import export_onnx_n1d7 as exp  # noqa: E402

log = logging.getLogger("build_trt_n17")


def _state_dims_from_checkpoint(model_path: str, embodiment_tag: str) -> dict:
    """Read per-state-key dim from ``<model>/experiment_cfg/dataset_statistics.json``.

    The model checkpoint ships the exact state shapes the model was trained on,
    keyed by embodiment tag. Use that as the ground truth instead of guessing.
    """
    stats = json.loads((Path(model_path) / "experiment_cfg/dataset_statistics.json").read_text())
    return {k: len(v["mean"]) for k, v in stats[embodiment_tag]["state"].items()}


def _synthetic_obs(policy, prompt: str, state_dims: dict, image_hw: tuple) -> dict:
    """Build a one-step obs matching the policy's modality config."""
    h, w = image_hw
    cfg = policy.get_modality_config()
    obs: dict = {}
    for k in cfg["video"].modality_keys:
        obs[f"video.{k}"] = np.zeros((1, h, w, 3), dtype=np.uint8)
    for k in cfg["state"].modality_keys:
        obs[f"state.{k}"] = np.zeros((1, state_dims[k]), dtype=np.float32)
    for k in cfg["language"].modality_keys:
        obs[k] = prompt
    return exp.parse_observation_gr00t(obs, cfg)


_PATCHED_STATE_DIMS: dict = {}
_PATCHED_IMAGE_HW: list = [(256, 256)]  # mutable holder; overwritten before patch fires


def _patch_dataset(prompt: str) -> None:
    """Replace the LeRobot dataset loader and ``prepare_observation`` so the
    upstream pipeline reads zero data from disk and grabs shapes from a
    synthetic observation instead.
    """

    class _StubDataset:
        def __init__(self, *_, **__):
            pass

        def __len__(self):
            return 1

        def __getitem__(self, _):
            return None

    exp.LeRobotEpisodeLoader = _StubDataset
    exp.prepare_observation = lambda policy, _ds, traj_idx=0: _synthetic_obs(
        policy, prompt, _PATCHED_STATE_DIMS, _PATCHED_IMAGE_HW[0]
    )


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", required=True, choices=SUPPORTED)
    args, forwarded = p.parse_known_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s: %(message)s")

    policy_cfg = get_environment_config(args.env)["policy"]
    model_path = snapshot_download(
        repo_id=policy_cfg["model_repo"],
        revision=policy_cfg.get("model_revision"),
        cache_dir=str(Path.home() / ".cache/soarm_models"),
    )
    prompt = policy_cfg.get("task_description") or policy_cfg.get("language_instruction") or ""
    embodiment_tag = policy_cfg.get("embodiment_tag", "new_embodiment")
    _PATCHED_STATE_DIMS.update(_state_dims_from_checkpoint(model_path, embodiment_tag))
    raw = policy_cfg.get("image_size") or [256, 256]
    _PATCHED_IMAGE_HW[0] = (int(raw[0]), int(raw[1]))
    log.info("state dims: %s | image_hw: %s", _PATCHED_STATE_DIMS, _PATCHED_IMAGE_HW[0])
    _patch_dataset(prompt)

    out_dir = REPO / "trt_engines" / args.env
    out_dir.mkdir(parents=True, exist_ok=True)

    cfg = build.PipelineConfig(
        model_path=model_path,
        dataset_path="/dev/null",
        output_dir=str(out_dir),
        embodiment_tag=policy_cfg.get("embodiment_tag", "new_embodiment"),
        video_backend="decord",
        steps="export,build",
    )
    log.info("Running build_trt_pipeline (synthetic obs, no dataset) ...")
    build.main(cfg)


if __name__ == "__main__":
    main()

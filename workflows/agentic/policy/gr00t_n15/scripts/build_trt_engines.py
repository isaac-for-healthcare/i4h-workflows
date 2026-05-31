#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Build TRT DiT engines for GR00T N1.5 policy envs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import torch
from common.config import get_environment_config
from huggingface_hub import snapshot_download

REPO = Path(__file__).resolve().parents[1]
AGENTIC = REPO.parents[1]
SUPPORTED = ("scissor_pick_and_place", "assemble_trocar")

log = logging.getLogger("build_trt_n15")


class DiTInputCapture:
    def __init__(self) -> None:
        self.captured = False
        self.sa_embs: torch.Tensor | None = None
        self.vl_embs: torch.Tensor | None = None
        self.timestep: torch.Tensor | None = None

    def hook_fn(self, _module, args, kwargs) -> None:
        if self.captured:
            return
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        encoder_hidden_states = kwargs["encoder_hidden_states"] if "encoder_hidden_states" in kwargs else args[1]
        timestep = kwargs["timestep"] if "timestep" in kwargs else args[2]
        self.sa_embs = hidden_states.detach().cpu().clone()
        self.vl_embs = encoder_hidden_states.detach().cpu().clone()
        self.timestep = timestep.detach().cpu().clone()
        self.captured = True
        log.info(
            "captured DiT inputs: sa_embs=%s vl_embs=%s timestep=%s",
            _tensor_shape(self.sa_embs),
            _tensor_shape(self.vl_embs),
            _tensor_shape(self.timestep),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=SUPPORTED)
    parser.add_argument("--workspace", type=int, default=8192, help="TensorRT workspace MB.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s: %(message)s")

    policy_cfg = get_environment_config(args.env)["policy"]
    prepare_normalized_input = None
    max_vl_seq_len = None
    if args.env == "assemble_trocar":
        import policy.assemble_trocar.infer.data_config  # noqa: F401
        from policy.assemble_trocar.infer.policy import _prepare_rl_normalized_input

        prepare_normalized_input = _prepare_rl_normalized_input
        max_vl_seq_len = 850

    from gr00t.data.embodiment_tags import EmbodimentTag
    from gr00t.experiment.data_config import DATA_CONFIG_MAP
    from gr00t.model.policy import Gr00tPolicy

    data_config = DATA_CONFIG_MAP[policy_cfg["data_config"]]
    if args.env == "scissor_pick_and_place":
        data_config.video_keys = ["video.room", "video.wrist"]
    model_path = snapshot_download(
        repo_id=policy_cfg["model_repo"],
        revision=policy_cfg.get("model_revision"),
        cache_dir=str(Path.home() / ".cache/rheo_models"),
    )
    log.info("loading policy from %s", model_path)
    policy = Gr00tPolicy(
        model_path=model_path,
        modality_config=data_config.modality_config(),
        modality_transform=data_config.transform(),
        embodiment_tag=EmbodimentTag(policy_cfg.get("embodiment_tag", "new_embodiment")),
    )

    image_hw = _image_hw(policy, fallback=_fallback_image_hw(policy_cfg, default=(480, 640)))
    log.info("synthetic image_hw=%s", image_hw)
    prompt = policy_cfg.get("task_description") or policy_cfg.get("language_instruction") or ""
    capture = _capture_dit_inputs(
        policy,
        _synthetic_obs(policy, prompt, image_hw),
        prepare_normalized_input=prepare_normalized_input,
    )

    out = REPO / "trt_engines" / args.env
    onnx_path = out / "onnx" / "dit_model.onnx"
    engine_path = out / "engines" / "dit_bf16.engine"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("exporting ONNX -> %s", onnx_path)
    _export_dit_to_onnx(policy, capture, onnx_path)

    shapes = {
        "sa_embs": _tensor_shape(capture.sa_embs),
        "vl_embs": _tensor_shape(capture.vl_embs),
        "timestep": _tensor_shape(capture.timestep),
    }
    log.info("building TensorRT engine -> %s", engine_path)
    _build_engine(onnx_path, engine_path, args.workspace, shapes, max_vl_seq_len=max_vl_seq_len)
    log.info("done")


def _capture_dit_inputs(policy, obs: dict[str, Any], *, prepare_normalized_input=None) -> DiTInputCapture:
    capture = DiTInputCapture()
    hook = policy.model.action_head.model.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)
    try:
        with torch.inference_mode():
            if prepare_normalized_input is None:
                policy.get_action(obs)
            else:
                normalized_input = policy.apply_transforms(obs)
                normalized_input = prepare_normalized_input(normalized_input)
                policy._get_action_from_normalized_input(normalized_input)
    finally:
        hook.remove()
    if not capture.captured:
        raise RuntimeError("DiT input capture failed; synthetic obs may not match the model modality config.")
    return capture


def _synthetic_obs(policy, prompt: str, image_hw: tuple[int, int]) -> dict[str, Any]:
    h, w = image_hw
    obs: dict[str, Any] = {}
    cfg = policy.get_modality_config()
    state_shapes = _metadata_shapes(policy)
    for key in cfg["video"].modality_keys:
        obs[key] = np.zeros((1, h, w, 3), dtype=np.uint8)
    for key in cfg["state"].modality_keys:
        obs[key] = np.zeros((1, state_shapes[_short_key(key)]), dtype=np.float32)
    for key in cfg["language"].modality_keys:
        obs[key] = prompt
    return obs


def _metadata_shapes(policy) -> dict[str, int]:
    state = getattr(getattr(policy.metadata, "modalities", None), "state", None) or {}
    shapes = {}
    for key, block in state.items() if isinstance(state, dict) else vars(state).items():
        shape = block.get("shape") if isinstance(block, dict) else getattr(block, "shape", None)
        if shape:
            shapes[key] = int(shape[0])
    return shapes


def _fallback_image_hw(policy_cfg: dict[str, Any], *, default: tuple[int, int]) -> tuple[int, int]:
    raw = policy_cfg.get("image_size") or list(default)
    return int(raw[0]), int(raw[1])


def _image_hw(policy, fallback: tuple[int, int]) -> tuple[int, int]:
    video = getattr(getattr(policy.metadata, "modalities", None), "video", None) or {}
    items = video.items() if isinstance(video, dict) else vars(video).items()
    for _key, block in items:
        resolution = block.get("resolution") if isinstance(block, dict) else getattr(block, "resolution", None)
        if resolution and len(resolution) >= 2:
            width, height = int(resolution[0]), int(resolution[1])
            return height, width
    return fallback


def _tensor_shape(tensor: torch.Tensor | None) -> tuple[int, ...]:
    if tensor is None:
        raise RuntimeError("DiT input capture is missing a required tensor.")
    return tuple(tensor.shape)


def _short_key(key: str) -> str:
    return key.split(".", 1)[1] if "." in key else key


def _export_dit_to_onnx(policy, captured: DiTInputCapture, output_path: Path) -> None:
    dit = policy.model.action_head.model.to(torch.bfloat16).cuda().eval()
    sa_embs = torch.randn(_tensor_shape(captured.sa_embs), dtype=torch.bfloat16, device="cuda")
    vl_embs = torch.randn(_tensor_shape(captured.vl_embs), dtype=torch.bfloat16, device="cuda")
    timestep = torch.ones(_tensor_shape(captured.timestep), dtype=torch.int64, device="cuda")

    class DiTWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, sa_embs, vl_embs, timestep):
            return self.module(hidden_states=sa_embs, encoder_hidden_states=vl_embs, timestep=timestep)

    with torch.inference_mode():
        torch.onnx.export(
            DiTWrapper(dit),
            (sa_embs, vl_embs, timestep),
            str(output_path),
            input_names=["sa_embs", "vl_embs", "timestep"],
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes={"vl_embs": {1: "vl_seq_len"}},
            dynamo=False,
        )


def _build_engine(
    onnx_path: Path,
    engine_path: Path,
    workspace_mb: int,
    shapes: dict[str, tuple],
    *,
    max_vl_seq_len: int | None = None,
) -> None:
    deployment = AGENTIC / "third_party" / "Isaac-GR00T-1.7" / "scripts" / "deployment"
    if str(deployment) not in sys.path:
        sys.path.insert(0, str(deployment))
    import build_tensorrt_engine

    min_shapes, opt_shapes, max_shapes = _shape_profiles(shapes, max_vl_seq_len=max_vl_seq_len)
    build_tensorrt_engine.build_engine(
        onnx_path=str(onnx_path),
        engine_path=str(engine_path),
        precision="bf16",
        workspace_mb=workspace_mb,
        min_shapes=min_shapes,
        opt_shapes=opt_shapes,
        max_shapes=max_shapes,
    )


def _shape_profiles(
    shapes: dict[str, tuple[int, ...]],
    *,
    max_vl_seq_len: int | None = None,
) -> tuple[dict[str, tuple[int, ...]], dict[str, tuple[int, ...]], dict[str, tuple[int, ...]]]:
    min_shapes = dict(shapes)
    opt_shapes = dict(shapes)
    max_shapes = dict(shapes)
    if max_vl_seq_len is not None and "vl_embs" in shapes:
        vl_shape = list(shapes["vl_embs"])
        vl_shape[1] = max(max_vl_seq_len, int(vl_shape[1]))
        max_shapes["vl_embs"] = tuple(vl_shape)
    return min_shapes, opt_shapes, max_shapes


if __name__ == "__main__":
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    main()

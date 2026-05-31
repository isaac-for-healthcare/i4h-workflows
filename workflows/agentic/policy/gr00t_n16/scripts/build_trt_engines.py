#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# pyright: reportMissingImports=false, reportMissingModuleSource=false
"""Build TRT DiT engines for GR00T N1.6 loco-manip policy envs.

Usage:
    build_trt_engines.py --env locomanip_tray_pick_and_place
    build_trt_engines.py --env locomanip_push_cart
"""

from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path
from typing import Any

import torch
from common.config import get_environment_config
from policy.registry import known_env_ids

REPO = Path(__file__).resolve().parents[1]
AGENTIC = REPO.parents[1]
SUPPORTED = known_env_ids()

log = logging.getLogger("build_trt_n16")


class DiTInputCapture:
    def __init__(self) -> None:
        self.captured = False
        self.sa_embs: torch.Tensor | None = None
        self.vl_embs: torch.Tensor | None = None
        self.timestep: torch.Tensor | None = None
        self.image_mask: torch.Tensor | None = None
        self.backbone_attention_mask: torch.Tensor | None = None

    def hook_fn(self, _module, args, kwargs) -> None:
        if self.captured:
            return
        hidden_states = kwargs["hidden_states"] if "hidden_states" in kwargs else args[0]
        encoder_hidden_states = kwargs["encoder_hidden_states"] if "encoder_hidden_states" in kwargs else args[1]
        timestep = kwargs["timestep"] if "timestep" in kwargs else args[2]
        image_mask = kwargs.get("image_mask")
        backbone_attention_mask = kwargs.get("backbone_attention_mask")
        self.sa_embs = hidden_states.detach().cpu().clone()
        self.vl_embs = encoder_hidden_states.detach().cpu().clone()
        self.timestep = timestep.detach().cpu().clone()
        self.image_mask = image_mask.detach().cpu().clone() if image_mask is not None else None
        self.backbone_attention_mask = (
            backbone_attention_mask.detach().cpu().clone() if backbone_attention_mask is not None else None
        )
        self.captured = True
        log.info(
            "captured DiT inputs: sa_embs=%s vl_embs=%s timestep=%s image_mask=%s backbone_attention_mask=%s",
            _tensor_shape(self.sa_embs),
            _tensor_shape(self.vl_embs),
            _tensor_shape(self.timestep),
            _tensor_shape(self.image_mask),
            _tensor_shape(self.backbone_attention_mask),
        )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--env", required=True, choices=SUPPORTED)
    parser.add_argument("--workspace", type=int, default=8192, help="TensorRT workspace MB.")
    parser.add_argument("--max-vl-seq-len", type=int, default=850, help="Max dynamic VLM sequence length.")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="[%(asctime)s] %(name)s: %(message)s")

    policy_cfg = get_environment_config(args.env)["policy"]
    policy_cfg = dict(policy_cfg)
    policy_cfg["trt_engine_path"] = None

    from policy.locomanip.infer.closedloop_policy import G1LocomanipClosedloopPolicy

    log.info(
        "loading policy: env=%s model=%s revision=%s",
        args.env,
        policy_cfg["model_repo"],
        policy_cfg.get("model_revision"),
    )
    policy = G1LocomanipClosedloopPolicy(
        num_envs=1,
        device="cuda",
        model_repo=policy_cfg["model_repo"],
        model_revision=policy_cfg.get("model_revision"),
        policy_config_data=policy_cfg,
        config_base_dir=AGENTIC,
    )

    image_hw = _fallback_image_hw(policy_cfg, default=(480, 640))
    log.info("synthetic image_hw=%s", image_hw)
    capture = _capture_dit_inputs(policy, _synthetic_observation(policy, image_hw))

    out = REPO / "trt_engines" / args.env
    onnx_path = out / "onnx" / "dit_model.onnx"
    engine_path = out / "engines" / "dit_bf16.engine"
    onnx_path.parent.mkdir(parents=True, exist_ok=True)
    engine_path.parent.mkdir(parents=True, exist_ok=True)

    log.info("exporting ONNX -> %s", onnx_path)
    _export_dit_to_onnx(policy.policy, capture, onnx_path)

    shapes = {
        "sa_embs": _tensor_shape(capture.sa_embs),
        "vl_embs": _tensor_shape(capture.vl_embs),
        "timestep": _tensor_shape(capture.timestep),
        "image_mask": _tensor_shape(capture.image_mask),
        "backbone_attention_mask": _tensor_shape(capture.backbone_attention_mask),
    }
    log.info("building TensorRT engine -> %s", engine_path)
    _build_engine(onnx_path, engine_path, args.workspace, shapes, max_vl_seq_len=args.max_vl_seq_len)
    log.info("done")


def _capture_dit_inputs(policy, obs: dict[str, Any]) -> DiTInputCapture:
    capture = DiTInputCapture()
    hook = policy.policy.model.action_head.model.register_forward_pre_hook(capture.hook_fn, with_kwargs=True)
    try:
        with torch.inference_mode():
            policy.policy.get_action(policy.get_observations(obs))
    finally:
        hook.remove()
    if not capture.captured:
        raise RuntimeError("DiT input capture failed; synthetic obs may not match the model modality config.")
    return capture


def _synthetic_observation(policy, image_hw: tuple[int, int]) -> dict[str, Any]:
    h, w = image_hw
    camera_obs = {}
    for spec in getattr(policy.policy_config, "pov_cam_names_sim", None) or ():
        obs_key = spec["obs_key"] if isinstance(spec, dict) else spec[0]
        camera_obs.setdefault(obs_key, torch.zeros((1, h, w, 3), dtype=torch.uint8))
    if not camera_obs:
        raise ValueError("policy.pov_cam_names_sim must define at least one camera mapping")
    return {
        "camera_obs": camera_obs,
        "policy": {
            "robot_joint_pos": torch.zeros((1, 43), dtype=torch.float32),
        },
    }


def _fallback_image_hw(policy_cfg: dict[str, Any], *, default: tuple[int, int]) -> tuple[int, int]:
    raw = policy_cfg.get("image_size") or list(default)
    return int(raw[0]), int(raw[1])


def _tensor_shape(tensor: torch.Tensor | None) -> tuple[int, ...]:
    if tensor is None:
        raise RuntimeError("DiT input capture is missing a required tensor.")
    return tuple(tensor.shape)


def _export_dit_to_onnx(policy, captured: DiTInputCapture, output_path: Path) -> None:
    dit = policy.model.action_head.model.to(torch.bfloat16).cuda().eval()
    sa_embs = torch.randn(_tensor_shape(captured.sa_embs), dtype=torch.bfloat16, device="cuda")
    vl_embs = torch.randn(_tensor_shape(captured.vl_embs), dtype=torch.bfloat16, device="cuda")
    timestep = torch.ones(_tensor_shape(captured.timestep), dtype=torch.int64, device="cuda")
    image_mask = _clone_to_cuda(captured.image_mask, dtype=torch.bool)
    backbone_attention_mask = _clone_to_cuda(captured.backbone_attention_mask, dtype=torch.bool)

    class DiTWrapper(torch.nn.Module):
        def __init__(self, module):
            super().__init__()
            self.module = module

        def forward(self, sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask):
            return self.module(
                hidden_states=sa_embs,
                encoder_hidden_states=vl_embs,
                timestep=timestep,
                image_mask=image_mask,
                backbone_attention_mask=backbone_attention_mask,
            )

    with torch.inference_mode():
        torch.onnx.export(
            DiTWrapper(dit),
            (sa_embs, vl_embs, timestep, image_mask, backbone_attention_mask),
            str(output_path),
            input_names=["sa_embs", "vl_embs", "timestep", "image_mask", "backbone_attention_mask"],
            output_names=["output"],
            opset_version=19,
            do_constant_folding=True,
            export_params=True,
            dynamic_axes={
                "vl_embs": {1: "vl_seq_len"},
                "image_mask": {1: "vl_seq_len"},
                "backbone_attention_mask": {1: "vl_seq_len"},
            },
            dynamo=False,
        )


def _clone_to_cuda(tensor: torch.Tensor | None, *, dtype: torch.dtype) -> torch.Tensor:
    if tensor is None:
        raise RuntimeError("DiT input capture is missing a required mask tensor.")
    return tensor.to(device="cuda", dtype=dtype).contiguous()


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
    if max_vl_seq_len is not None:
        for name in ("vl_embs", "image_mask", "backbone_attention_mask"):
            if name not in shapes:
                continue
            vl_shape = list(shapes[name])
            vl_shape[1] = max(max_vl_seq_len, int(vl_shape[1]))
            max_shapes[name] = tuple(vl_shape)
    return min_shapes, opt_shapes, max_shapes


if __name__ == "__main__":
    os.environ.setdefault("NO_ALBUMENTATIONS_UPDATE", "1")
    main()

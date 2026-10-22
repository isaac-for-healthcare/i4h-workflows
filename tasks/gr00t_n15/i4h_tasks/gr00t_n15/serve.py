# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Serve an i4h GR00T N1.5 checkpoint with GR00T 1.5's own ``RobotInferenceServer``.

The stock ``scripts/inference_service.py`` cannot load these checkpoints: the Trocar
checkpoint needs its own data config and inference preprocessing, and the scissors
checkpoint names cameras its stock data config does not. This module builds each policy
and serves it on GR00T's ZMQ protocol. Requests and replies carry torch
tensors rather than numpy arrays, because the simulator's numpy 2 pickles do not load
under this environment's numpy 1; the bridge converts at the boundary.

    tasks/gr00t_n15/.venv/bin/python -m i4h_tasks.gr00t_n15.serve assemble_trocar --port 5555
"""

from __future__ import annotations

import argparse
import logging
from typing import Any

import numpy as np
import torch

logger = logging.getLogger("i4h_tasks.gr00t_n15.serve")

#: Tasks served here; each one's checkpoint is pinned in manifest/<task>.yaml.
TASKS = ("assemble_trocar", "scissor_pick_and_place")


class TensorBridge:
    """A GR00T policy whose requests and replies cross the wire as tensors."""

    def __init__(self, policy: Any) -> None:
        self._policy = policy

    def get_action(self, observations: dict[str, Any]) -> dict[str, Any]:
        inputs = {key: value.numpy() if torch.is_tensor(value) else value for key, value in observations.items()}
        actions = self._policy.get_action(inputs)
        return {key: torch.from_numpy(np.ascontiguousarray(value)) for key, value in actions.items()}

    def get_modality_config(self) -> dict[str, Any]:
        return self._policy.get_modality_config()


def load(task: str, model_path: str | None, *, device: str) -> Any:
    from i4h_common.training import task_spec

    spec = task_spec(f"gr00t_n15/{task}")
    model = spec.model
    if task == "assemble_trocar":
        from i4h_tasks.gr00t_n15.assemble_trocar.infer.policy import load_assemble_trocar_policy, resolve_model_path

        path = resolve_model_path(model_path, model.get("repo"), model.get("revision"))
        return load_assemble_trocar_policy(
            path, device=device, action_head_future_tokens=int(model.get("action_head_future_tokens", 32))
        )
    if task == "scissor_pick_and_place":
        from gr00t.experiment.data_config import DATA_CONFIG_MAP
        from gr00t.model.policy import Gr00tPolicy

        data_config = DATA_CONFIG_MAP[str(model["data_config"])]
        # The stock data_config names the cameras it was written for (so100_dualcam says
        # video.front); the checkpoint was trained on the scene's, and loading checks them.
        cameras = list(spec.observation.get("cameras") or ())
        if cameras:
            data_config.video_keys = [f"video.{name}" for name in cameras]
        return Gr00tPolicy(
            model_path=model_path or str(model["repo"]),
            modality_config=data_config.modality_config(),
            modality_transform=data_config.transform(),
            embodiment_tag=str(model.get("embodiment_tag", "new_embodiment")),
            denoising_steps=int(model.get("denoising_steps", 4)),
            # Without this the head builds future_tokens from scratch and the checkpoint's
            # weights are silently replaced by random ones (NaN on the first inference).
            action_head_future_tokens=int(model.get("action_head_future_tokens", 0)),
            device=device,
        )
    raise ValueError(f"unknown task {task!r}; known: {list(TASKS)}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="i4h_tasks.gr00t_n15.serve", description=__doc__.split("\n\n")[0])
    parser.add_argument("task", choices=TASKS)
    parser.add_argument("--model-path", help="local checkpoint; default: the pinned Hugging Face revision")
    parser.add_argument("--host", default="*")
    parser.add_argument("--port", type=int, default=5555)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args(argv)
    logging.basicConfig(level=logging.INFO)

    from gr00t.eval.robot import RobotInferenceServer

    policy = load(args.task, args.model_path, device=args.device)
    RobotInferenceServer(TensorBridge(policy), host=args.host, port=args.port).run()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""PI0 policy runner (openpi-backed) for the Franka ultrasound liver scan.

Mirrors ``workflows/robotic_ultrasound/scripts/policy/pi0/runners.py`` but
exposes the same ``infer(observation) -> np.ndarray`` shape the agentic
policy services use.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

import numpy as np

logger = logging.getLogger("policy")


@dataclass
class RunnerConfig:
    model_path: str
    """HF repo id or local checkpoint path."""
    repo_id: str = "i4h/sim_liver_scan"
    """LeRobot repo id used for dataset normalization stats."""
    task_description: str = "Perform a liver ultrasound."


class PI0PolicyRunner:
    def __init__(self, cfg: RunnerConfig):
        self.cfg = cfg
        self._model = None

    def ensure_loaded(self) -> None:
        if self._model is not None:
            return
        # openpi pulls in JAX/Flax — heavy imports kept lazy.
        from openpi.policies import policy_config
        from policy.ultrasound_liver_scan.config import get_config

        config = get_config(name="robotic_ultrasound", repo_id=self.cfg.repo_id)
        logger.info("Loading PI0 model from %s ...", self.cfg.model_path)
        t0 = time.monotonic()
        self._model = policy_config.create_trained_policy(config, self.cfg.model_path)
        logger.info("PI0 policy ready in %.1fs", time.monotonic() - t0)

    def infer(self, observation: dict, num_steps: Optional[int] = None) -> np.ndarray:
        self.ensure_loaded()
        from openpi_client import image_tools

        room = image_tools.convert_to_uint8(image_tools.resize_with_pad(observation["room"], 224, 224))
        wrist = image_tools.convert_to_uint8(image_tools.resize_with_pad(observation["wrist"], 224, 224))
        joint_pos = np.asarray(observation["joint_positions"], dtype=np.float64)

        element = {
            "observation/image": room,
            "observation/wrist_image": wrist,
            "observation/state": joint_pos[:7],
            "prompt": self.cfg.task_description,
        }
        actions = np.asarray(self._model.infer(element)["actions"], dtype=np.float64)
        if num_steps is not None:
            actions = actions[:num_steps]
        return actions

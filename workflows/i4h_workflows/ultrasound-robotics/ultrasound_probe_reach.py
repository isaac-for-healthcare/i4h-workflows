# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Author, train, and run a torso-surface ultrasound probe reach."""

from i4h_engine.interface import Workflow
from i4h_tasks.basic.predicates import env_success
from i4h_workflow_modes.idle import idle
from i4h_workflow_modes.policy import policy

WORKFLOW = Workflow(
    scene="ultrasound_probe_reach",
    modes={
        "idle": idle,
        "policy": lambda: policy("rsl_rl/ultrasound_probe_reach", until=env_success),
    },
)

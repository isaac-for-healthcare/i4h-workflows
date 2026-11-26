# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""i4h controllers as IsaacLab-Arena policies.

Each module registers its policies with Arena on import, so Arena's runner loads them by
class path::

    python -m isaaclab_arena.evaluation.policy_runner \\
        --policy_type i4h_arena.policies.reach.CartesianReachPolicy --num_episodes 3 \\
        --external_environment_class_path i4h_arena.environments:PsmReach psm_reach

Rule-based controllers follow Isaac Lab's scripted state machines
(``scripts/environments/state_machine``): a per-env phase advanced once per step from
privileged scene state, reset per env when the env resets. The env's own termination
terms decide success and time-out.
"""

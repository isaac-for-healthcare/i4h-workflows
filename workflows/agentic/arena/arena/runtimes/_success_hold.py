# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Success-hold wrapper for episode termination.

Many of the humanoid pick/place success checks (object within X cm of target)
trigger transiently as the object passes the goal pose. The runtime treats
any ``terminated`` flag as episode-complete, so a fluke single-step proximity
ends the episode before the task actually finishes.

This helper wraps a ``success`` termination term so the underlying condition
must hold for ``hold_steps`` consecutive env steps before the wrapper returns
True. Mirrors ``workflows/rheo_0.5/scripts/utils/policy_tasks.py``.
"""

from __future__ import annotations

import torch


def create_success_hold_wrapper(original_success_term, hold_steps: int, num_envs: int, verbose: bool = False):
    """Wrap ``original_success_term`` so success must hold for ``hold_steps`` env steps."""
    from isaaclab.managers import TerminationTermCfg

    state = {
        "success_counters": [0] * num_envs,
        "success_achieved": [False] * num_envs,
        "last_episode_step": [0] * num_envs,
    }

    def success_hold_checker(env, **kwargs):
        for env_idx in range(num_envs):
            current_step = env.episode_length_buf[env_idx].item() if hasattr(env, "episode_length_buf") else 0
            # Reset per-env counters on episode boundary.
            if current_step < state["last_episode_step"][env_idx] or current_step == 0:
                state["success_counters"][env_idx] = 0
                state["success_achieved"][env_idx] = False
            state["last_episode_step"][env_idx] = current_step

        original_results = original_success_term.func(env, **kwargs)
        held_results = torch.zeros_like(original_results, dtype=torch.bool)

        for env_idx in range(len(original_results)):
            if state["success_achieved"][env_idx]:
                held_results[env_idx] = True
                continue
            if original_results[env_idx]:
                state["success_counters"][env_idx] += 1
                if state["success_counters"][env_idx] >= hold_steps:
                    held_results[env_idx] = True
                    state["success_achieved"][env_idx] = True
                    if verbose:
                        print(f"[success-hold] env {env_idx}: held for {hold_steps} steps")
            else:
                state["success_counters"][env_idx] = 0

        return held_results

    return TerminationTermCfg(
        func=success_hold_checker,
        params=original_success_term.params.copy() if original_success_term.params else {},
        time_out=getattr(original_success_term, "time_out", False),
    )


def apply_success_hold(env, hold_steps: int, num_envs: int) -> bool:
    """Patch the env's ``success`` termination term with the hold wrapper.

    Returns True if a success term was found and wrapped, False otherwise.
    Call once after ``env.reset()`` and before running episodes.
    """
    if hold_steps <= 1:
        return False
    if not (hasattr(env.cfg, "terminations") and hasattr(env.cfg.terminations, "success")):
        return False

    wrapped = create_success_hold_wrapper(env.cfg.terminations.success, hold_steps, num_envs)
    env.cfg.terminations.success = wrapped
    if hasattr(env, "termination_manager"):
        tm = env.termination_manager
        if hasattr(tm, "_term_name_to_term_idx") and "success" in tm._term_name_to_term_idx:
            tm._term_cfgs[tm._term_name_to_term_idx["success"]] = wrapped
            return True
    return True

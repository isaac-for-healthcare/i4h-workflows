# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

from __future__ import annotations

import logging

import h5py
import numpy as np
import torch
from arena.runtimes.core.base import ready

logger = logging.getLogger("arena")


def _load_state_tree(group, device: str) -> dict:
    return {
        kind: {
            asset: {
                field: torch.as_tensor(np.array(group[kind][asset][field]), device=device)
                for field in group[kind][asset].keys()
            }
            for asset in group[kind].keys()
        }
        for kind in group.keys()
    }


def _state_frame_count(demo) -> int:
    if "states" not in demo:
        return 0
    states = demo["states"]
    for kind in states.keys():
        for asset in states[kind].keys():
            for field in states[kind][asset].keys():
                return int(states[kind][asset][field].shape[0])
    return 0


def _state_at(trajectory: dict, index: int) -> dict:
    return {
        kind: {
            asset: {field: values[index : index + 1] for field, values in fields.items()}
            for asset, fields in assets.items()
        }
        for kind, assets in trajectory.items()
    }


def _demo_action_key(demo) -> str | None:
    if "actions" in demo:
        return "actions"
    if "obs/actions" in demo:
        return "obs/actions"
    return None


def _demo_replay_frame_count(demo) -> int:
    action_key = _demo_action_key(demo)
    action_count = 0 if action_key is None else int(demo[action_key].shape[0])
    return max(_state_frame_count(demo), action_count)


@torch.no_grad()
def run_recorded_episode(ctx, *, dataset_path: str, episode_index: int) -> None:
    with h5py.File(dataset_path, "r") as f:
        demos = [name for name in f["data"].keys() if _demo_replay_frame_count(f["data"][name]) > 0]
        if not demos:
            raise ValueError(f"no replayable demos with states or actions found in {dataset_path}")
        if episode_index >= len(demos):
            raise ValueError(f"episode_index {episode_index} out of range (have {len(demos)} replayable demos)")
        demo = f["data"][demos[episode_index]]
        demo_name = demos[episode_index]
        state_count = _state_frame_count(demo)
        if state_count > 0:
            state_trajectory = _load_state_tree(demo["states"], ctx.device)
            actions = None
            initial_state = None
        else:
            action_key = _demo_action_key(demo)
            if action_key is None:
                raise ValueError(f"demo {demo_name} has no states or actions dataset")
            actions = np.array(demo[action_key], dtype=np.float32)
            initial_state = _load_state_tree(demo["initial_state"], ctx.device)
            state_trajectory = None

    if state_trajectory is not None:
        logger.info("replay episode %s: %s state frames", demo_name, state_count)
        for frame_idx in range(state_count):
            if not ready(ctx):
                return
            ctx.env.scene.reset_to(_state_at(state_trajectory, frame_idx), env_ids=None, is_relative=True)
            ctx.env.sim.forward()
            ctx.env.sim.render()
        return

    logger.info("replay episode %s: %s actions", demo_name, len(actions))
    ctx.env.reset_to(initial_state, env_ids=None, is_relative=True)
    for action in actions:
        if not ready(ctx):
            return "aborted"
        ctx.env.step(torch.tensor(action, device=ctx.device).repeat(ctx.env.unwrapped.num_envs, 1))
    return "completed"

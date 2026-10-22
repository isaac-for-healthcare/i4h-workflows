# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""The scene state an episode starts from, captured for recording and restored for replay.

A recording's actions only reproduce the episode from the state its reset
drew. Scenes randomize that state -- the liver-scan phantom moves up to 15 cm
and turns up to 90 degrees -- so a replay that resets normally applies the
recorded probe sweep to a phantom somewhere else. The recorder stores env 0's
post-reset state beside the actions, and replay resets to it instead of
drawing a new one.

The state is IsaacLab's ``InteractiveScene.get_state(is_relative=True)``,
converted to numpy with the batch kept at one env so the HDF5 layout matches
IsaacLab's own ``initial_state``.
"""

from __future__ import annotations

import logging
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import h5py
import numpy as np
import torch

from i4h_common.episode import DATASET_FORMAT_VERSION, dataset_format_version, demo_names, read_initial_state

logger = logging.getLogger("i4h_arena.recording")


def capture_scene_state(env: Any) -> dict[str, Any] | None:
    """Env 0's scene state as numpy, or ``None`` when the env cannot report one."""
    scene = getattr(getattr(env, "unwrapped", env), "scene", None)
    get_state = getattr(scene, "get_state", None)
    if not callable(get_state):
        return None
    try:
        return _to_numpy(get_state(is_relative=True))
    except Exception:  # a backend without full state support must not stop the recording
        logger.warning(
            "could not capture the initial scene state; this episode will replay from a fresh reset", exc_info=True
        )
        return None


def _to_numpy(tree: Mapping[str, Any]) -> dict[str, Any]:
    return {
        key: _to_numpy(value) if isinstance(value, Mapping) else value[:1].detach().cpu().numpy()
        for key, value in tree.items()
    }


def restore_scene_state(env: Any, state: Mapping[str, Any], *, seed: int | None = None) -> None:
    """Reset every env to a recorded state instead of the scene's randomized reset.

    ``reset_to`` still runs the reset events, then overwrites every asset the
    state names, so anything the state does not cover (materials, lighting) is
    reset as usual.
    """
    unwrapped = getattr(env, "unwrapped", env)
    current = unwrapped.scene.get_state(is_relative=True)
    missing = [
        f"{family}/{name}" for family, assets in current.items() for name in assets if name not in state.get(family, {})
    ]
    if missing:
        raise ValueError(
            f"the recorded initial state has no entry for {missing}; "
            "it was recorded from a different scene configuration"
        )
    tensors = {
        family: {name: _to_torch(state[family][name], unwrapped.num_envs, unwrapped.device) for name in assets}
        for family, assets in current.items()
    }
    unwrapped.reset_to(tensors, None, seed=seed, is_relative=True)


def _to_torch(value: Any, num_envs: int, device: Any) -> Any:
    if isinstance(value, Mapping):
        return {key: _to_torch(item, num_envs, device) for key, item in value.items()}
    array = torch.as_tensor(np.asarray(value), device=device)
    return array.repeat(num_envs, *([1] * (array.ndim - 1)))


def load_initial_state(dataset: str | Path, episode: int, scene: str) -> dict[str, Any] | None:
    """The initial state one recorded episode started from, if it applies to ``scene``.

    ``None``, with a warning, when the episode has no stored state, was
    recorded in a different scene, or is in a file that declares a dataset
    format other than the XYZW one ``reset_to`` expects. Asset names alone
    cannot tell scenes apart (``psm_needle`` and ``psm_block`` both call their
    manipulandum ``object``), so the recording's scene is compared by name. A
    file that declares no format is an i4h recording and is XYZW. The replay
    then starts from the scene's own reset, as it did before states were
    recorded.
    """
    with h5py.File(str(dataset), "r") as handle:
        names = demo_names(handle)
        if episode >= len(names):
            return None
        demo = handle["data"][names[episode]]
        recorded_scene = demo.attrs.get("scene", handle["data"].attrs.get("scene"))
        state = read_initial_state(demo)
        format_version = dataset_format_version(handle)
    if isinstance(recorded_scene, bytes):
        recorded_scene = recorded_scene.decode()
    if state is None:
        logger.warning(
            "%s episode %s has no initial scene state (recorded before it was stored); "
            "the scene resets with its own randomization, so objects may not be where they were recorded",
            dataset,
            episode,
        )
        return None
    if format_version != DATASET_FORMAT_VERSION:
        logger.warning(
            "%s declares Isaac Lab dataset format_version %s, not %s; its root poses may not be XYZW, so episode %s's "
            "initial state is not applied and the scene resets with its own randomization",
            dataset,
            format_version,
            DATASET_FORMAT_VERSION,
            episode,
        )
        return None
    if recorded_scene and recorded_scene != scene:
        logger.warning(
            "%s episode %s was recorded in scene %r, not %r; its initial state is not applied "
            "and the scene resets with its own randomization",
            dataset,
            episode,
            recorded_scene,
            scene,
        )
        return None
    return state

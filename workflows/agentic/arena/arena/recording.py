# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Episode recording for the manip envs.

Streaming HDF5 writer wired through IsaacLab's :class:`RecorderManager`. Two
modes:

* **filter_success=True (default):** only episodes that the caller explicitly
  marks successful via :func:`save_successful_episode` are exported. Failed
  attempts must be cleared with :func:`discard_episode` before the next
  attempt. The buffered HDF5 never contains partial / failed demos.
* **filter_success=False:** every buffered episode is exported on demand via
  :func:`save_episode`. Failed episodes can still be saved if the caller
  chooses (e.g. for offline diagnostics).

The dataset path comes from ``--record-to``.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import torch

logger = logging.getLogger("arena")


def _term_false(env):
    return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def _term_true(env):
    return torch.ones(env.num_envs, dtype=torch.bool, device=env.device)


def setup_recording(
    ctx,
    dataset_path: str,
    *,
    streaming: bool = True,
    filter_success: bool = True,
) -> None:
    """Install a :class:`RecorderManager` on the env and remember the path.

    Args:
        ctx: SimpleNamespace with at least ``env``.
        dataset_path: HDF5 output path; parent dirs are created.
        streaming: Use :class:`StreamingRecorderManager` so the file grows
            during an episode. Filtered recording forces this off because
            discarded attempts must never be partially exported.
        filter_success: When True, only :func:`save_successful_episode` exports
            an episode; failed attempts must be cleared via
            :func:`discard_episode`. When False, every buffered episode flushes
            on :func:`save_episode`.
    """
    from isaaclab.managers import DatasetExportMode, RecorderManager, TerminationTermCfg
    from isaaclab.utils.datasets import HDF5DatasetFileHandler

    env = ctx.env
    path = Path(dataset_path).expanduser().resolve()
    path.parent.mkdir(parents=True, exist_ok=True)
    if streaming and filter_success:
        logger.info("recording filter_success=True; disabling streaming so discarded attempts stay out of HDF5")
        streaming = False

    cfg = env.cfg.recorders
    cfg.dataset_export_mode = (
        DatasetExportMode.EXPORT_ALL
        if streaming
        else (DatasetExportMode.EXPORT_SUCCEEDED_ONLY if filter_success else DatasetExportMode.EXPORT_ALL)
    )
    cfg.dataset_export_dir_path, cfg.dataset_filename = str(path.parent), path.stem
    cfg.dataset_file_handler_class_type = HDF5DatasetFileHandler
    ctx.recording_dataset_path = str(_hdf5_path(path))
    ctx.recording_filter_success = bool(filter_success)

    if filter_success:
        env.cfg.terminations.success = TerminationTermCfg(func=_term_false)
        if "success" in env.termination_manager.active_terms:
            _set_success_term(env, _term_false)
        else:
            _ensure_success_term(env, TerminationTermCfg(func=_term_false))
        if hasattr(env.cfg.terminations, "time_out") and env.cfg.terminations.time_out is not None:
            env.cfg.terminations.time_out = None
            if "time_out" in env.termination_manager.active_terms:
                _remove_term(env.termination_manager, "time_out")

    if not hasattr(env, "_original_recorder_manager"):
        env._original_recorder_manager = env.recorder_manager
    del env.recorder_manager
    env.recorder_manager = _recorder_manager(RecorderManager, cfg, env, streaming)


@torch.no_grad()
def mark_success(ctx) -> None:
    """Flip the recording success term to True (so record_pre_reset exports)."""
    _set_success_term(ctx.env, _term_true)
    ctx.env.termination_manager.compute()


@torch.no_grad()
def reset_success(ctx) -> None:
    """Flip the recording success term back to False between attempts."""
    _set_success_term(ctx.env, _term_false)


@torch.no_grad()
def save_successful_episode(ctx, metadata: dict[str, Any] | None = None) -> None:
    """Flush the current episode's buffered steps as a successful demo.

    Marks the envs as successful (so the recorder writes them) and exports.
    Safe to call in either filter mode.
    """
    env = ctx.env
    if getattr(ctx, "recording_filter_success", False):
        mark_success(ctx)
    env.recorder_manager.set_success_to_episodes(None, torch.ones(env.num_envs, dtype=torch.bool, device=env.device))
    env.recorder_manager.export_episodes()
    if metadata:
        _append_episode_metadata(ctx, metadata)
    _prune_active_incomplete_demos(ctx)
    if getattr(ctx, "recording_filter_success", False):
        reset_success(ctx)


@torch.no_grad()
def save_episode(ctx, metadata: dict[str, Any] | None = None) -> None:
    """Flush the current episode's buffered steps to disk.

    Alias of :func:`save_successful_episode`. Kept for callers that don't
    distinguish success from failure (save-all mode).
    """
    save_successful_episode(ctx, metadata=metadata)


def discard_episode(ctx) -> None:
    """Drop the current buffered episode without exporting.

    Use after a failed attempt in filter mode so the next attempt starts with
    a clean buffer.
    """
    rm = getattr(ctx.env, "recorder_manager", None)
    if rm is None:
        return
    clear = getattr(rm, "_clear_episode_cache", None)
    if callable(clear):
        clear()
    else:
        # Fallback for non-streaming recorder managers: reset() clears state.
        rm.reset()


def close_recording(ctx) -> None:
    """Close the HDF5 file, restore the original recorder, and (optionally)
    drop any incomplete demos that the streaming recorder wrote.

    In filter-success mode the streaming recorder writes a partial demo
    (often just ``initial_state``) on the next ``env.reset()`` after a failed
    attempt, before our in-memory ``discard_episode`` can intervene. The
    final pass here deletes those incomplete demos from the closed HDF5 so
    downstream tools see only successful episodes.
    """
    rm = getattr(ctx.env, "recorder_manager", None)
    if rm is not None and (fh := getattr(rm, "_dataset_file_handler", None)) is not None:
        fh.close()
    if (orig := getattr(ctx.env, "_original_recorder_manager", None)) is not None:
        ctx.env.recorder_manager = orig
    if getattr(ctx, "recording_filter_success", False):
        dataset_path = getattr(ctx, "recording_dataset_path", None)
        if dataset_path:
            _prune_incomplete_demos(dataset_path, drop_unsuccessful=True)


def _prune_incomplete_demos(dataset_path: str, *, drop_unsuccessful: bool = False) -> None:
    """Drop incomplete, and optionally unsuccessful, demos from the HDF5 in place."""
    try:
        import h5py
    except ImportError:
        logger.warning("h5py not available; skipping incomplete-demo prune")
        return

    try:
        with h5py.File(dataset_path, "a") as h5_file:
            to_drop = _drop_incomplete_demos(h5_file, drop_unsuccessful=drop_unsuccessful)
            if to_drop:
                logger.info(
                    "dropped %s incomplete/unsuccessful demo(s) from %s: %s", len(to_drop), dataset_path, to_drop
                )
    except Exception:
        logger.exception("failed to prune incomplete demos from %s", dataset_path)


def _has_actions(demo) -> bool:
    if "actions" in demo:
        return True
    obs = demo.get("obs")
    return obs is not None and hasattr(obs, "keys") and "actions" in obs


def _prune_active_incomplete_demos(ctx) -> None:
    h5_file = _active_hdf5_file(ctx)
    if h5_file is None:
        return
    try:
        to_drop = _drop_incomplete_demos(
            h5_file,
            drop_unsuccessful=bool(getattr(ctx, "recording_filter_success", False)),
        )
        if to_drop:
            logger.info("dropped %s active incomplete/unsuccessful demo(s): %s", len(to_drop), to_drop)
            h5_file.flush()
    except Exception:
        logger.exception("failed to prune active incomplete demos")


def _drop_incomplete_demos(h5_file, *, drop_unsuccessful: bool = False) -> list[str]:
    data = h5_file.get("data")
    if data is None:
        return []
    to_drop = []
    for name in list(data.keys()):
        demo = data.get(name)
        if demo is None or not hasattr(demo, "keys"):
            continue
        if not _has_actions(demo) or (drop_unsuccessful and not _demo_successful(demo)):
            to_drop.append(name)
    for name in to_drop:
        del data[name]
    return to_drop


def _demo_successful(demo) -> bool:
    value = demo.attrs.get("success", False)
    if hasattr(value, "item"):
        value = value.item()
    return bool(value)


def _set_success_term(env, func) -> None:
    from isaaclab.managers import TerminationTermCfg

    env.termination_manager.set_term_cfg("success", TerminationTermCfg(func=func))


def _ensure_success_term(env, term_cfg) -> None:
    tm = env.termination_manager
    if "success" in tm.active_terms:
        return
    tm._resolve_common_term_cfg("success", term_cfg, min_argc=1)
    tm._term_names.append("success")
    tm._term_cfgs.append(term_cfg)
    tm._term_name_to_term_idx["success"] = len(tm._term_names) - 1
    new_col = torch.zeros((env.num_envs, 1), device=env.device, dtype=torch.bool)
    tm._term_dones = torch.cat([tm._term_dones, new_col], dim=1)
    tm._last_episode_dones = torch.cat([tm._last_episode_dones, new_col], dim=1)


def _remove_term(tm, name: str) -> None:
    if name not in tm._term_name_to_term_idx:
        return
    idx = tm._term_name_to_term_idx.pop(name)
    tm._term_names.pop(idx)
    tm._term_cfgs.pop(idx)
    # Rebuild the index map since indices shifted after pop.
    tm._term_name_to_term_idx = {n: i for i, n in enumerate(tm._term_names)}
    # Drop the matching column from the done buffers.
    keep = [i for i in range(tm._term_dones.shape[1]) if i != idx]
    tm._term_dones = tm._term_dones[:, keep] if keep else tm._term_dones[:, :0]
    tm._last_episode_dones = tm._last_episode_dones[:, keep] if keep else tm._last_episode_dones[:, :0]


def _recorder_manager(manager_cls, cfg, env, streaming: bool):
    if not streaming:
        return manager_cls(cfg, env)

    from leisaac.enhance.managers import StreamingRecorderManager

    manager = StreamingRecorderManager(cfg, env)
    manager.flush_steps = 100
    manager.compression = "lzf"
    return manager


def _hdf5_path(path: Path) -> Path:
    return path if path.suffix in {".h5", ".hdf5"} else path.with_suffix(".hdf5")


def _append_episode_metadata(ctx, metadata: dict[str, Any]) -> None:
    h5_file = _active_hdf5_file(ctx)
    try:
        if h5_file is not None:
            _write_metadata_to_latest_episode(h5_file, metadata)
            h5_file.flush()
            return
    except Exception:
        logger.exception("recording metadata skipped: failed to update active HDF5 file")
        return

    dataset_path = getattr(ctx, "recording_dataset_path", None)
    if not dataset_path:
        logger.warning("recording metadata skipped: dataset path is unavailable")
        return

    try:
        import h5py

        with h5py.File(dataset_path, "a") as h5_file:
            _write_metadata_to_latest_episode(h5_file, metadata)
    except Exception:
        logger.exception("recording metadata skipped: failed to update %s", dataset_path)


def _active_hdf5_file(ctx):
    rm = getattr(ctx.env, "recorder_manager", None)
    fh = getattr(rm, "_dataset_file_handler", None)
    if fh is None:
        return None
    for attr in ("_hdf5_file_stream", "_hdf5_file", "_h5_file", "_file"):
        candidate = getattr(fh, attr, None)
        if candidate is not None and hasattr(candidate, "require_group") and hasattr(candidate, "flush"):
            return candidate
    return None


def _write_metadata_to_latest_episode(h5_file, metadata: dict[str, Any]) -> None:
    episode_group = _latest_episode_group(h5_file, require_actions=True) or _latest_episode_group(h5_file)
    if episode_group is None:
        logger.warning("recording metadata skipped: no episode group found")
        return

    safe_metadata = _json_safe(metadata)
    episode_group.attrs["agentic_metadata_json"] = json.dumps(safe_metadata, sort_keys=True)
    for key in ("env_id", "run_id", "episode_index", "attempt_index", "status"):
        value = safe_metadata.get(key)
        if value is not None:
            episode_group.attrs[f"agentic_{key}"] = value


def _latest_episode_group(h5_file, *, require_actions: bool = False):
    for base in (h5_file.get("data"), h5_file):
        if base is None:
            continue
        groups = [
            base[name]
            for name in base.keys()
            if _is_episode_group_name(name)
            and hasattr(base[name], "keys")
            and hasattr(base[name], "attrs")
            and (not require_actions or _has_actions(base[name]))
        ]
        if groups:
            return sorted(groups, key=_episode_sort_key)[-1]
    return None


def _is_episode_group_name(name: str) -> bool:
    return name.startswith(("demo", "episode"))


def _episode_sort_key(group) -> tuple[int, str]:
    name = group.name.rsplit("/", 1)[-1]
    suffix = name.rsplit("_", 1)[-1]
    try:
        return int(suffix), name
    except ValueError:
        return -1, name


def _json_safe(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _json_safe(inner) for key, inner in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(inner) for inner in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if hasattr(value, "item"):
        try:
            return value.item()
        except ValueError:
            pass
    if hasattr(value, "tolist"):
        return value.tolist()
    return str(value)

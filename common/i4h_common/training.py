# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Training defaults, read from the task manifest.

Finetuning hyper-parameters live in the same ``model`` / ``train`` blocks the
inference server reads, preventing training and serving configuration drift.

This module is in ``i4h_common`` because all four policy stacks need it and none of
them can import each other.
"""

from __future__ import annotations

import logging
from typing import Any

logger = logging.getLogger("i4h.training")


def task_spec(task_id: str) -> Any:
    """Look up a task manifest entry. Raises with a readable message if absent.

    Scans manifests directly rather than going through
    ``i4h_engine.registry``: ``i4h_common`` sits below ``i4h_engine`` in the
    layering and must not reach upward, even lazily. The scan is the same
    filesystem glob, and the parser is already here.
    """
    from i4h_common.paths import workflow_root  # noqa: PLC0415
    from i4h_common.taskdef import load_taskdef  # noqa: PLC0415

    project, _, name = task_id.partition("/")
    if not name:
        raise KeyError(f"task id must be '<project>/<name>', got {task_id!r}")
    # One file per task, named for it — so the lookup is a path, not a scan.
    manifest = workflow_root() / "tasks" / project / "i4h_tasks" / project / "manifest" / f"{name}.yaml"
    if not manifest.is_file():
        raise KeyError(f"no manifest at {manifest} for task {task_id!r}")
    return load_taskdef(manifest)[2]


def _spec_or_none(task_id: str) -> Any:
    try:
        return task_spec(task_id)
    except Exception:  # noqa: BLE001 - training must still run against a bare dataset
        logger.warning("no manifest entry for %s; falling back to built-in defaults", task_id)
        return None


def task_default(task_id: str, key: str, fallback: Any, catalog: dict | None = None) -> Any:
    """A value from the task's ``model`` entry in its stack catalog.

    Reading the *same* entry the inference server reads is what keeps a
    checkpoint's training and serving configuration from drifting apart.
    """
    if catalog is None:
        spec = _spec_or_none(task_id)
        value = None if spec is None else spec.model.get(key)
    else:
        entry = catalog.get(task_id.rpartition("/")[2], {})
        value = (entry.get("model") or {}).get(key)
    return fallback if value is None else value


def train_default(task_id: str, key: str, fallback: Any, catalog: dict | None = None) -> Any:
    """A value from the task's ``train`` entry in its stack catalog.

    ``catalog`` is passed in by the stack's ``train.py`` — this module cannot
    import it, since each stack lives in its own venv.
    """
    if catalog is None:
        spec = _spec_or_none(task_id)
        value = None if spec is None else spec.train.get(key)
    else:
        entry = catalog.get(task_id.rpartition("/")[2], {})
        value = (entry.get("train") or {}).get(key)
    return fallback if value is None else value


def require_trainable(task_id: str) -> Any:
    """Return the spec, or explain why this task cannot be finetuned.

    Plenty of tasks are inference-only: a released checkpoint that nobody
    retrains locally. Saying so plainly beats letting the trainer start and
    fail somewhere inside the data loader.
    """
    spec = task_spec(task_id)
    if not spec.trainable:
        from i4h_common.paths import workflow_root  # noqa: PLC0415

        project, _, name = task_id.partition("/")
        manifest = workflow_root() / "tasks" / project / "i4h_tasks" / project / "manifest" / f"{name}.yaml"
        raise SystemExit(
            f"{task_id} is inference-only: no train block in {manifest}.\n"
            f"It serves a released checkpoint ({spec.model.get('repo', 'unknown')}). "
            "Add a train block to make it finetunable."
        )
    return spec


def default_base_model(task_id: str, fallback: str) -> str:
    """Base checkpoint to finetune from.

    ``[task.train].base_model`` if present, else the manifest's serving ``repo``
    — finetuning the model you are already serving is the common case.
    """
    spec = _spec_or_none(task_id)
    if spec is None:
        return fallback
    return str(spec.train.get("base_model") or spec.model.get("repo") or fallback)


def resolve_dataset(path_or_repo_id: str) -> str:
    """Accept a local LeRobot directory or a HuggingFace repo id."""
    from pathlib import Path  # noqa: PLC0415

    candidate = Path(path_or_repo_id).expanduser()
    if candidate.exists():
        return str(candidate.resolve())
    if "/" in path_or_repo_id and not path_or_repo_id.startswith("/"):
        from huggingface_hub import snapshot_download  # noqa: PLC0415

        logger.info("downloading dataset %s", path_or_repo_id)
        return snapshot_download(repo_id=path_or_repo_id, repo_type="dataset")
    raise FileNotFoundError(
        f"no dataset at {path_or_repo_id!r}. Convert a recording first:\n"
        f"  uv run --project tools/dataset i4h-dataset convert demos.hdf5 <out> --robot <name>"
    )

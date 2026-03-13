# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Apply / revert the GR00T RL policy patch via ``git apply``.

RLinf modifies the GR00T model during RL post-training (eagle-input padding
and dropout removal).  When evaluating an RL checkpoint with the standalone
eval script we need the same modifications applied to
``gr00t/model/policy.py``.

This module wraps ``git apply`` / ``git checkout`` around the patch file at:

    tools/env_setup/patches/gr00t_policy_padding_dropout.patch

Usage as a context manager (auto-reverts on exit)::

    from policy.apply_gr00t_rl_patch import gr00t_rl_patch

    with gr00t_rl_patch():
        from gr00t.model.policy import Gr00tPolicy
        policy = Gr00tPolicy(...)
        # ... run evaluation ...
    # patch is automatically reverted here

Or apply / revert manually::

    from policy.apply_gr00t_rl_patch import apply_patch, revert_patch

    apply_patch()
    # ... do work ...
    revert_patch()
"""

import logging
import subprocess
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

# Resolve paths relative to the repo root.
# Expected layout:
#   <REPO_ROOT>/third_party/Isaac-GR00T/          ← git apply target
#   <REPO_ROOT>/tools/env_setup/patches/gr00t_policy_padding_dropout.patch
#
# This file lives at:
#   <REPO_ROOT>/workflows/rheo/scripts/policy/apply_gr00t_rl_patch.py
_THIS_DIR = Path(__file__).resolve().parent
_REPO_ROOT = _THIS_DIR.parents[3]  # workflows/rheo/scripts/policy -> repo root
_PATCH_FILE = _REPO_ROOT / "tools" / "env_setup" / "patches" / "gr00t_policy_padding_dropout.patch"
_GROOT_DIR = _REPO_ROOT / "third_party" / "Isaac-GR00T"

# Files touched by the patch (used for git checkout restore).
_PATCHED_FILES = ["gr00t/model/policy.py"]


def _is_patch_applied() -> bool:
    """Check whether the patch has already been applied."""
    result = subprocess.run(
        ["git", "apply", "--check", "-R", str(_PATCH_FILE)],
        cwd=str(_GROOT_DIR),
        capture_output=True,
    )
    # If reverse-apply check succeeds, the patch is currently applied.
    return result.returncode == 0


def _restore_files_to_head() -> None:
    """Restore the patched files to their git HEAD state."""
    subprocess.run(
        ["git", "checkout", "HEAD", "--"] + _PATCHED_FILES,
        cwd=str(_GROOT_DIR),
        check=True,
    )


def apply_patch() -> None:
    """Apply the GR00T RL policy patch (idempotent).

    First restores the target files to their committed (HEAD) state so
    that ``git apply`` always finds the expected context, regardless of
    any prior manual edits.
    """
    if not _PATCH_FILE.exists():
        raise FileNotFoundError(f"Patch file not found: {_PATCH_FILE}")

    if _is_patch_applied():
        logger.info("GR00T RL patch already applied, skipping.")
        print("[apply_gr00t_rl_patch] Patch already applied, skipping.")
        return

    # Restore to HEAD first so the patch context always matches.
    logger.info("Restoring patched files to HEAD before applying patch...")
    _restore_files_to_head()

    logger.info("Applying GR00T RL patch: %s", _PATCH_FILE)
    print(f"[apply_gr00t_rl_patch] Applying patch: {_PATCH_FILE}")
    subprocess.run(
        ["git", "apply", str(_PATCH_FILE)],
        cwd=str(_GROOT_DIR),
        check=True,
    )
    logger.info("GR00T RL patch applied successfully.")
    print("[apply_gr00t_rl_patch] Patch applied successfully.")


def revert_patch() -> None:
    """Revert the GR00T RL policy patch by restoring files to HEAD."""
    if not _is_patch_applied():
        logger.info("GR00T RL patch not applied, nothing to revert.")
        return

    logger.info("Reverting GR00T RL patch (restoring to HEAD)...")
    print("[apply_gr00t_rl_patch] Reverting patch (restoring to HEAD)...")
    _restore_files_to_head()
    logger.info("GR00T RL patch reverted successfully.")
    print("[apply_gr00t_rl_patch] Patch reverted successfully.")


@contextmanager
def gr00t_rl_patch():
    """Context manager that applies the patch on enter and reverts on exit."""
    apply_patch()
    try:
        yield
    finally:
        revert_patch()

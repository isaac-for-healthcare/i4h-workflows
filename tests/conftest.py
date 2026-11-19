# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Fixtures shared by the repository-level tests."""

from __future__ import annotations

from collections.abc import Iterator

import pytest


@pytest.fixture(autouse=True, scope="session")
def _private_pid_dir(tmp_path_factory: pytest.TempPathFactory) -> Iterator[None]:
    """Keep the launches under test out of the checkout's .run/, where real runs track their backends."""
    with pytest.MonkeyPatch.context() as patch:
        patch.setenv("I4H_PID_DIR", str(tmp_path_factory.mktemp("pids.d")))
        yield

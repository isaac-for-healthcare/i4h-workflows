# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Hold to the end of a fixed-length episode.

Workflows that must run a fixed number of steps — so an evaluation episode is
the same length as its reference recording — end with one of these rather
than stopping the moment success is detected."""

from __future__ import annotations

from i4h_tasks.basic.control.wait import Wait


class Hold(Wait):
    """Alias of :class:`Wait` that reads better as a terminal node.

    Workflows that must run a fixed number of steps — so an evaluation episode is
    the same length as its reference recording — end with one of these rather
    than stopping the moment success is detected.
    """

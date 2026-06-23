# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tagged stderr logging helpers.

These prints survive the very chatty `omni.rtx` / Kit logging streams so that
short, high-signal traces (e.g. one-line dumps of MDP-term state) stay visible
in long terminal sessions.
"""

from __future__ import annotations

import sys


def tagged_log(tag: str, msg: str) -> None:
    """Print ``[tag] msg`` to stderr, flushed immediately."""
    print(f"[{tag}] {msg}", file=sys.stderr, flush=True)


def tagged_error(tag: str, msg: str) -> None:
    """Print ``[tag][ERROR] msg`` to stderr, flushed immediately."""
    print(f"[{tag}][ERROR] {msg}", file=sys.stderr, flush=True)


def make_logger(tag: str):
    """Return ``(log, error)`` callables that auto-prefix with ``tag``."""

    def _log(msg: str) -> None:
        tagged_log(tag, msg)

    def _error(msg: str) -> None:
        tagged_error(tag, msg)

    return _log, _error

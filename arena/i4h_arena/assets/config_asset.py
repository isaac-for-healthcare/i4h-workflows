# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Arena wrapper for an existing Isaac Lab manager configuration."""

from __future__ import annotations

from typing import Any

from isaaclab_arena.assets.asset import Asset


class ConfigAsset(Asset):
    """Expose an already-built manager config through Arena's asset contract."""

    def __init__(self, name: str, cfg: Any, tags: list[str] | None = None) -> None:
        super().__init__(name=name, tags=tags or ["scene"])
        self._cfg = cfg

    def get_object_cfg(self) -> tuple[str, Any]:
        return self.name, self._cfg

    def get_event_cfg(self) -> tuple[str, None]:
        return self.name, None

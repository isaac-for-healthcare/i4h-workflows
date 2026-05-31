# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Any


class JsonlWriter:
    def __init__(self, path: str | None) -> None:
        self._path = Path(path).expanduser() if path else None
        self._file = None

    def __enter__(self) -> JsonlWriter:
        if self._path:
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._file = self._path.open("a", encoding="utf-8")
        return self

    def __exit__(self, *exc_info) -> None:
        if self._file:
            self._file.close()

    def write(self, record: dict[str, Any]) -> None:
        line = json.dumps(record, ensure_ascii=True)
        if self._file:
            self._file.write(line + "\n")
            self._file.flush()
        print(line, file=sys.stdout, flush=True)

#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Return code-ready facts for an i4h coding agent without launching Isaac Sim."""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict
from pathlib import Path
from typing import Any

WORKFLOWS_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(WORKFLOWS_ROOT / "arena"))

from i4h_arena.assets.authoring_catalog import AUTHORING_ASSETS, authoring_asset
from i4h_arena.authoring import load_snapshot, manifest_capabilities


def _asset_payload(name: str) -> dict[str, Any]:
    return {"name": name, **asdict(authoring_asset(name))}


def _resolved_snapshot(path: Path, workflow: str) -> dict[str, Any]:
    snapshot = load_snapshot(path, workflow)
    items: list[dict[str, Any]] = []
    for source in snapshot["items"]:
        item = dict(source)
        if item["kind"] == "known_asset":
            item["catalog"] = asdict(authoring_asset(item["preset"]))
        items.append(item)
    return {
        "snapshot": {**snapshot, "items": items},
        "manifest_capabilities": manifest_capabilities(snapshot),
    }


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compact", action="store_true", help="emit compact JSON")
    commands = parser.add_subparsers(dest="command", required=True)

    commands.add_parser("assets", help="list every reusable asset preset")

    asset = commands.add_parser("asset", help="show one reusable asset preset")
    asset.add_argument("name", choices=sorted(AUTHORING_ASSETS))

    snapshot = commands.add_parser(
        "snapshot",
        help="validate a live snapshot and resolve its catalog and manifest facts",
    )
    snapshot.add_argument("workflow")
    snapshot.add_argument("path", type=Path)
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "assets":
        payload: object = [_asset_payload(name) for name in sorted(AUTHORING_ASSETS)]
    elif args.command == "asset":
        payload = _asset_payload(args.name)
    else:
        payload = _resolved_snapshot(args.path, args.workflow)
    print(
        json.dumps(
            payload,
            indent=None if args.compact else 2,
            separators=(",", ":") if args.compact else None,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

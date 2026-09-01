# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""``list`` / ``show`` / ``lint`` — the three commands that need no simulator.

``run.sh`` dispatches to this for those verbs, so they answer in milliseconds
from the light venv rather than after a Kit launch.
"""

from __future__ import annotations

import argparse
import sys

from i4h_engine.lint import lint_workflow
from i4h_engine.loader import available_workflows, load_workflow_module, resolve_workflow
from i4h_engine.registry import default_registry
from i4h_engine.render import to_mermaid, to_text


def _add_workflow_arg(parser: argparse.ArgumentParser, *, optional: bool = False) -> None:
    parser.add_argument("workflow", nargs="?" if optional else None, help="workflow name (see `list`)")
    parser.add_argument("--mode", default=None, help="workflow run mode; defaults to idle")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="i4h-workflow", description="Inspect and validate i4h workflows.")
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list workflows, tasks and scenes")
    listing.add_argument("--tasks", action="store_true", help="list registered tasks instead of workflows")
    listing.add_argument("--scenes", action="store_true", help="list registered scenes instead of workflows")

    show = sub.add_parser("show", help="render the selected TaskGraph")
    _add_workflow_arg(show)
    show.add_argument("--format", choices=("text", "mermaid"), default="text")

    lint = sub.add_parser("lint", help="validate a workflow against the registry and its scene")
    _add_workflow_arg(lint, optional=True)
    lint.add_argument("--all", action="store_true", help="lint every workflow and every run mode")
    return parser


def _cmd_list(args: argparse.Namespace) -> int:
    registry = default_registry()
    if args.tasks:
        for task_id in sorted(registry.tasks):
            spec = registry.tasks[task_id]
            trainable = "trainable" if spec.trainable else ""
            print(f"{task_id:<44} {spec.runtime:<10} {trainable:<10} {spec.summary}")
        return 0
    if args.scenes:
        for name in sorted(registry.scenes):
            scene = registry.scenes[name]
            print(f"{name:<24} {scene.embodiment:<12} {scene.action_space:<16} dof={scene.dof}")
        return 0

    names = available_workflows()
    if not names:
        print("no workflows found under workflows/i4h_workflows/<specialty>/", file=sys.stderr)
        return 1
    for name in names:
        try:
            workflow_module = load_workflow_module(name)
            modes = ",".join(sorted(workflow_module.modes))
            print(f"{name:<32} scene={workflow_module.scene:<22} modes={modes}")
        except Exception as exc:  # noqa: BLE001 - one broken module must not hide the rest
            print(f"{name:<32} !! {type(exc).__name__}: {exc}")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    workflow = resolve_workflow(args.workflow, args.mode)
    renderers = {"text": to_text, "mermaid": to_mermaid}
    print(renderers[args.format](workflow))
    return 0


def _cmd_lint(args: argparse.Namespace) -> int:
    registry = default_registry()
    targets: list[tuple[str, str | None]] = []
    if getattr(args, "all", False):
        for name in available_workflows():
            workflow_module = load_workflow_module(name)
            targets.extend((name, mode) for mode in sorted(workflow_module.modes))
    else:
        if not args.workflow:
            raise ValueError("lint requires a workflow name or --all")
        targets.append((args.workflow, args.mode))

    failed = 0
    for name, mode in targets:
        workflow = resolve_workflow(name, mode)
        report = lint_workflow(workflow, registry)
        print(f"{name} [{workflow.mode}]")
        print(report.render())
        if not report.ok:
            failed += 1
    return 1 if failed else 0


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handlers = {"list": _cmd_list, "show": _cmd_show, "lint": _cmd_lint}
    try:
        return handlers[args.command](args)
    except (KeyError, AttributeError, TypeError, ValueError) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

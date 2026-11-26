# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Manager term functions must not default the arguments Isaac Lab passes positionally.

Isaac Lab 3 copies every defaulted parameter of a term function into the term's
params, so a reset event declared ``def term(env, env_ids=None)`` is then called
with ``env_ids`` twice and fails at the first reset.
"""

from __future__ import annotations

import ast
from pathlib import Path

ARENA = Path(__file__).resolve().parents[1] / "i4h_arena"


def _term_env_ids(function: ast.FunctionDef) -> ast.arg | None:
    """The ``env_ids`` argument of a term shaped ``f(env, env_ids, ...)`` or ``__call__(self, env, env_ids, ...)``."""
    names = [arg.arg for arg in function.args.args]
    if names[:2] == ["env", "env_ids"]:
        return function.args.args[1]
    if function.name == "__call__" and names[:3] == ["self", "env", "env_ids"]:
        return function.args.args[2]
    return None


def test_no_term_function_defaults_env_ids():
    offenders = []
    for path in sorted(ARENA.rglob("*.py")):
        for node in ast.walk(ast.parse(path.read_text(), filename=str(path))):
            if not isinstance(node, ast.FunctionDef) or (arg := _term_env_ids(node)) is None:
                continue
            args = node.args.args
            if args.index(arg) >= len(args) - len(node.args.defaults):
                offenders.append(f"{path.relative_to(ARENA.parent)}:{node.lineno} {node.name}")

    assert not offenders, "term functions default env_ids:\n" + "\n".join(offenders)

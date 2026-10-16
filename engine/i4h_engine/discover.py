# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Finding the tasks that exist.

**One mechanism**: every task project declares its tasks as YAML under
``manifest/``, one file per task, named for it. Discovery is a single glob —
``tasks/*/i4h_tasks/*/manifest/*.yaml`` — with no branching on what kind of task it
is. That uniformity is worth more than tailoring the mechanism per case: a
reader learns one rule, and ``run.sh list`` works with nothing installed.

What differs is only how much each file has to say, and that follows from
whether the declaration has anywhere better to live:

* **in-process** (``basic``, ``ik``, ``teleop``) — the manifest holds ``summary``,
  optional ``prompt``, and ``impl``. The class holds behavior-derived requirements
  and ``Inputs``/``Outputs`` ports, read on demand by :meth:`TaskSpec.resolve`.
* **remote** (``gr00t_*``, ``openpi_pi0``) — everything, because nothing can
  import those stacks. Their torch pins conflict with Isaac's, which is the
  whole reason ``RemoteTask`` exists.

The runtime is not declared — ``impl`` decides it. A task with a class to point
at runs in-process; one without is served over the bus. Asking for both would
let them contradict each other.

So the manifest always answers "what tasks are there, and where does each one
live"; the class answers the rest when it can.
"""

from __future__ import annotations

import logging
from pathlib import Path

from i4h_common.manifest import BackendSpec, TaskSpec
from i4h_common.taskdef import TaskDef, TaskDefError, load_taskdef

logger = logging.getLogger("i4h.discover")

#: Where every task project declares its tasks, relative to the workflow root.
MANIFEST_GLOB = "tasks/*/i4h_tasks/*/manifest/*.yaml"


def _spec(path: Path, runtime: str, definition: TaskDef, impl: str | None) -> TaskSpec:
    """Add what the location knows to what the author declared."""
    project = path.parent.parent.name
    backend = (
        BackendSpec(
            project=f"tasks/{project}",
            entry=definition.backend_entry or f"i4h_tasks.{project}.server:main",
        )
        if runtime == "remote"
        else None
    )
    return TaskSpec(
        project=project,
        name=path.stem,
        runtime=runtime,
        summary=definition.summary,
        prompt=definition.prompt,
        impl=impl,
        backend=backend,
        inputs={},
        outputs={"success": "bool"} if runtime == "remote" else {},
        requires=definition.merged_requires(),
        observation=dict(definition.observation),
        model=dict(definition.model),
        pre=dict(definition.precondition),
        post=dict(definition.postcondition),
        source=path,
        _trainable=definition.trainable,
        # Remote specs are complete as written; in-process ones still need the
        # class read for ports and requires.
        _resolved=runtime == "remote",
    )


def discover_tasks(root: Path) -> tuple[dict[str, TaskSpec], list[str]]:
    """Every declared task, plus any problems worth reporting."""
    found: dict[str, TaskSpec] = {}
    errors: list[str] = []
    for path in sorted(root.glob(MANIFEST_GLOB)):
        if ".venv" in path.parts:
            continue
        try:
            runtime, impl, definition = load_taskdef(path)
        except TaskDefError as exc:
            errors.append(str(exc))
            continue
        spec = _spec(path, runtime, definition, impl)
        if spec.id in found:
            errors.append(f"{path}: duplicate task id {spec.id!r} (already from {found[spec.id].source})")
            continue
        found[spec.id] = spec
    return found, errors

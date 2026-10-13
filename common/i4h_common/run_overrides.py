# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Which ``I4H_*`` overrides were in force, for the run that they shaped.

Several components read behaviour out of the environment: the catheter alone
has seventeen ``I4H_CATHETER_*`` variables covering damping, segment count,
stiffness, containment and the arrival tolerance. None of them were written
anywhere, so two runs of the same profile could differ in physics and in what
counted as success with nothing on disk to tell them apart.

The arrival tolerance is the one that makes this more than a reproducibility
nicety. It decides the ``success`` attribute on a recorded episode, that
attribute is what selects demonstrations for training, and the tolerance it was
measured at outlives the shell that set it. Two of this workflow's recordings
were labelled successful at 8 mm against a 5 mm default, and the only surviving
trace of a tolerance anywhere in ``runs/`` is a directory somebody hand-named
``..._tol7``.

Collected by prefix rather than from a declared list of names. A list is a
second place to update, and the one guarantee worth having here is that a knob
added later is captured without anybody remembering to come back. The cost is
that the raw string is recorded rather than the resolved value, so an override
a resolver rejected is recorded as the run saw it, not as it took effect --
which is the honest direction for provenance: it says what was asked for.
"""

from __future__ import annotations

import os
from collections.abc import Mapping

#: Set by ``run.sh`` to wire the run together, not to change its behaviour.
#: Recorded verbatim elsewhere in the same metadata or irrelevant to it.
LAUNCHER_VARIABLES = frozenset(
    {
        "I4H_RECORD_PATH",
        "I4H_RUN_DIR",
        "I4H_RUN_METADATA",
        "I4H_SETUP_PROJECTS",
        "I4H_THIRD_PARTY_TARGET",
        "I4H_VENV_ROOT",
        "I4H_WORKFLOWS",
        "I4H_WORKFLOWS_REPO_URL",
    }
)

OVERRIDE_PREFIX = "I4H_"


def environment_overrides(environ: Mapping[str, str] | None = None) -> dict[str, str]:
    """Set ``I4H_*`` variables that change behaviour, sorted by name.

    Empty when nothing is overridden, which is the common case and reads
    correctly in metadata as "nothing was overridden" rather than as a field
    somebody forgot to fill in.
    """
    source = os.environ if environ is None else environ
    return {
        name: str(value)
        for name, value in sorted(source.items())
        if name.startswith(OVERRIDE_PREFIX) and name not in LAUNCHER_VARIABLES and str(value).strip()
    }


__all__ = ["LAUNCHER_VARIABLES", "OVERRIDE_PREFIX", "environment_overrides"]

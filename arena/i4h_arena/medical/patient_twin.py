# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Re-export of the twin manifest, which now lives in :mod:`i4h_common.patient_twin`.

Moved because dataset conversion needs it too. The five goal columns the catheter
descriptor declares are derived rather than recorded, and deriving them means
resolving the same route the Scene built -- so the twin loader has to be readable
from ``tools/dataset``, which cannot import arena and its Isaac dependencies.

``i4h-common`` is where that belongs: it is the Isaac-free contract package both
already depend on, and this module needs only numpy and yaml. Re-exported from
here rather than moved outright so the arena call sites stay as they are.
"""

from __future__ import annotations

from i4h_common.patient_twin import PatientTwin

__all__ = ["PatientTwin"]

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
# SPDX-License-Identifier: Apache-2.0
"""Dependency-light navigation backend selection and recording provenance."""

from __future__ import annotations

import json
from pathlib import Path

CONTINUATION_SCHEMA = 1
PROVENANCE_KEYS = (
    "catheter_backend",
    "numerical_sim_revision",
    "continuation_schema_version",
    "patient_twin",
    "patient_twin_sha256",
    "fluoro_device",
)


def recording_provenance(path):
    import h5py

    path = Path(path)
    native = {}
    if path.is_file():
        with h5py.File(path, "r") as handle:
            native = json.loads(handle["data"].attrs.get("env_args", "{}"))
    sidecar = path.parent / "run.json"
    external = json.loads(sidecar.read_text()) if sidecar.is_file() else {}
    for key in PROVENANCE_KEYS:
        if key in native and key in external and native[key] != external[key]:
            raise ValueError(f"conflicting recording provenance: {key}")
    return {key: native.get(key, external.get(key)) for key in PROVENANCE_KEYS if key in native or key in external}


def resolve_backend(requested=None, recording=None):
    if requested not in (None, "numerical", "legacy"):
        raise ValueError(f"unknown catheter backend {requested!r}")
    if recording is None:
        return requested or "legacy"
    metadata = recording_provenance(recording)
    recorded = metadata.get("catheter_backend")
    if recorded is None:
        if requested != "legacy":
            raise ValueError("recording has no catheter backend metadata; pass explicit --catheter-backend legacy")
        return "legacy"
    if recorded not in ("numerical", "legacy"):
        raise ValueError(f"unknown recorded catheter backend {recorded!r}")
    if requested is not None and requested != recorded:
        raise ValueError(f"requested catheter backend {requested} differs from recording backend {recorded}")
    if recorded == "numerical" and (
        metadata.get("continuation_schema_version") != CONTINUATION_SCHEMA or not metadata.get("numerical_sim_revision")
    ):
        raise ValueError("numerical recording has unsupported or incomplete continuation provenance")
    return recorded

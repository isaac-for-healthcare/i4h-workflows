#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

e2e_count_demos() {
  local hdf5_path="$1"
  uv --directory workflows/agentic/mimic run python - "$hdf5_path" <<'PYEOF' 2>/dev/null
import h5py
import sys

with h5py.File(sys.argv[1], "r") as f:
    print(len(f.get("data", {})))
PYEOF
}

e2e_write_sim_filter() {
  local src="$1"
  local dst="$2"

  uv --directory workflows/agentic/mimic run python - "$src" "$dst" <<'PYEOF'
import h5py
import sys

src, dst = sys.argv[1], sys.argv[2]
with h5py.File(src, "r") as s, h5py.File(dst, "w") as o:
    for key, value in s.attrs.items():
        o.attrs[key] = value
    source_data = s["data"]
    output_data = o.create_group("data")
    for key, value in source_data.attrs.items():
        output_data.attrs[key] = value

    kept = 0
    for name, group in source_data.items():
        if bool(group.attrs.get("success", False)):
            output_name = f"demo_{kept}"
            s.copy(group, output_data, name=output_name)
            output_data[output_name].attrs["source_demo"] = name
            output_data[output_name].attrs["annotation_success"] = True
            output_data[output_name].attrs["annotation_reasoning"] = "sim: HDF5 success=True attr"
            kept += 1

    output_data.attrs["total"] = kept
    o.attrs["filtered_from_hdf5"] = src
    o.attrs["filtered_source"] = "sim_success_attr"
    if kept == 0:
        print("[e2e] no demos with success=True in expanded HDF5", file=sys.stderr)
        sys.exit(2)

    print(f"[e2e] sim-label filter wrote {kept} demos to {dst}")
PYEOF
}

e2e_copy_hdf5() {
  local src="$1"
  local dst="$2"
  uv --directory workflows/agentic/mimic run python - "$src" "$dst" <<'PYEOF'
from pathlib import Path
import shutil
import sys

src, dst = map(Path, sys.argv[1:3])
dst.parent.mkdir(parents=True, exist_ok=True)
if src.resolve() != dst.resolve():
    shutil.copy2(src, dst)
print(f"[e2e] copied {src} -> {dst}")
PYEOF
}

e2e_filter_source() {
  local hdf5_path="$1"
  uv --directory workflows/agentic/mimic run python - "$hdf5_path" <<'PYEOF' 2>/dev/null || true
import h5py
import sys

with h5py.File(sys.argv[1], "r") as f:
    print(f.attrs.get("filtered_source", "vlm"))
PYEOF
}

e2e_set_filter_source() {
  local hdf5_path="$1"
  local source="$2"
  uv --directory workflows/agentic/mimic run python - "$hdf5_path" "$source" <<'PYEOF'
import h5py
import sys

with h5py.File(sys.argv[1], "a") as f:
    f.attrs["filtered_source"] = sys.argv[2]
PYEOF
}

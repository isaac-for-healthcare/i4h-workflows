#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
# Provision only the pinned source needed by CPU regressions; no model downloads.
set -euo pipefail
root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
revision=a4b6abe205d7942f45cf3e8843c3e72ce818729d
checkout="$root/third_party/RLinf-a4b6abe"
patch="$root/third_party/rlinf_training_failures.patch"
if [ ! -d "$checkout" ]; then
  mkdir -p "$checkout"
  git -C "$checkout" init
  git -C "$checkout" remote add origin https://github.com/RLinf/RLinf.git
  git -C "$checkout" fetch --depth 1 origin "$revision"
  git -C "$checkout" checkout --detach FETCH_HEAD
fi
[ "$(git -C "$checkout" rev-parse HEAD)" = "$revision" ] || {
  echo "RL CPU tests require RLinf $revision" >&2
  exit 1
}
if git -C "$checkout" apply --reverse --check "$patch" 2>/dev/null; then
  echo "RLinf CPU patch already applied"
else
  git -C "$checkout" apply --check "$patch"
  git -C "$checkout" apply "$patch"
fi

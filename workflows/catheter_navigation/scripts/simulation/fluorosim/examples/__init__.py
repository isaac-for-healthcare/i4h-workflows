# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at

# http://www.apache.org/licenses/LICENSE-2.0

# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Runnable example entry points for the Fluoroscopy Simulator.

These modules are the workflow's CLI entry points, mirroring the
``python -m ...`` convention used by the other i4h workflows:

    python -m fluorosim.examples.preprocess_ct --dicom /path/to/dicom --output-dir /tmp/cache
    python -m fluorosim.examples.render_drr --cache /tmp/cache --output drr.png

``render_drr`` also works with no input data at all by falling back to a
built-in synthetic phantom, so the default workflow run is self-contained.
"""

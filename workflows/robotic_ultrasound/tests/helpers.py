# SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

import os
from unittest import skipUnless


def requires_rti(func):
    RTI_AVAILABLE = bool(os.getenv("RTI_LICENSE_FILE") and os.path.exists(os.getenv("RTI_LICENSE_FILE")))
    return skipUnless(RTI_AVAILABLE, "RTI Connext DDS is not installed or license not found")(func)

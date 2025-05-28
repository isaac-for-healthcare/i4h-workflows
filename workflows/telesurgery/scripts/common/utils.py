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

import distutils
import os

import ntplib


def get_ntp_offset(
    ntp_server=os.environ.get("NTP_SERVER_HOST", "pool.ntp.org"),
    ntp_port=int(os.environ.get("NTP_SERVER_PORT", "123")),
    version=3,
):
    client = ntplib.NTPClient()
    try:
        response = client.request(ntp_server, port=ntp_port, version=version)
        print(f"NTP Sync offset time: {response.offset:.6f}")
        return response.offset
    except Exception as e:
        print(f"[WARN] NTP sync failed: {e}")
        return 0


def strtobool(s):
    return False if s is None else s if isinstance(s, bool) else bool(distutils.util.strtobool(s))

#!/bin/bash

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

ENV_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/" >/dev/null 2>&1 && pwd)"


# Host IP for Patient/Surgeon
export PATIENT_IP="${PATIENT_IP:-127.0.0.1}"
export SURGEON_IP="${SURGEON_IP:-127.0.0.1}"

export NDDS_DISCOVERY_PEERS="${PATIENT_IP},${SURGEON_IP}"

# RTI QOS Profile
export NDDS_QOS_PROFILES=$ENV_SCRIPT_DIR/dds/qos_profile.xml

# RTI License
if [ -z "${RTI_LICENSE_FILE}" ]; then
  export RTI_LICENSE_FILE=$ENV_SCRIPT_DIR/dds/rti_license.dat
fi

# Python Path
export PYTHONPATH=$ENV_SCRIPT_DIR:$PYTHONPATH
export LD_LIBRARY_PATH=$ENV_SCRIPT_DIR/holohub/operators/nvidia_video_codec/lib:$LD_LIBRARY_PATH
# Optional: NTP Server to capture time diff between 2 nodes

if [ ! -z "${NTP_SERVER_HOST}" ] && [ ! -z "${NTP_SERVER_PORT}" ]; then
  export NTP_SERVER_HOST="${NTP_SERVER_HOST}"
  export NTP_SERVER_PORT="${NTP_SERVER_PORT}"
fi

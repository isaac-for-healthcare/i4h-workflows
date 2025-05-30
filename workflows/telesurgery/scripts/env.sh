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

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/" >/dev/null 2>&1 && pwd)"

# RTI QOS Profile
export NDDS_QOS_PROFILES=$SCRIPT_DIR/dds/qos_profile.xml

# RTI Discovery Address
export NDDS_DISCOVERY_PEERS="surgeon IP address"

# RTI License
if [ -z "${RTI_LICENSE_FILE}" ]; then
  export RTI_LICENSE_FILE=$SCRIPT_DIR/dds/rti_license.dat
fi

# Python Path
export PYTHONPATH=$SCRIPT_DIR
export LD_LIBRARY_PATH=$SCRIPT_DIR/holohub/operators/nvidia_video_codec/lib:$LD_LIBRARY_PATH
# Optional: NTP Server to capture time diff between 2 nodes
# export NTP_SERVER_HOST="surgeon IP address"
# export NTP_SERVER_PORT=123

# Host IP for Patient/Surgeon
export PATIENT_IP="patient IP address"
export SURGEON_IP="surgeon IP address"


VIDEO_CODEC_FILE=/tmp/nv_video_codec.zip

if [ ! -d "$SCRIPT_DIR/holohub/operators/nvidia_video_codec/lib" ]; then
  echo "Downloading NVIDIA Video Codec Operators"
  curl -L -o $VIDEO_CODEC_FILE 'https://edge.urm.nvidia.com:443/artifactory/sw-holoscan-cli-generic/holohub/operators/nv_video_codec_0.2.zip'
  unzip -o $VIDEO_CODEC_FILE -d $SCRIPT_DIR/holohub/operators/nvidia_video_codec
  rm $VIDEO_CODEC_FILE
fi

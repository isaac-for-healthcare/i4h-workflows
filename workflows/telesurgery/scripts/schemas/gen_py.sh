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

if [ ! -f "${NDDSHOME}"/bin/rtiddsgen ]; then
  echo "Install RTI using APT and export "
  echo "https://community.rti.com/static/documentation/developers/get-started/apt-install.html"
  echo "export NDDSHOME=/opt/rti.com/`ls /opt/rti.com/ | tail -n1`"
  exit 1
fi

for filename in idl/*.idl; do
  # delete manually for any updates before overriding existing ones
  #filename_no_ext=$(basename "$filename" .idl)
  #rm -rf ./"${filename_no_ext}".py
  "${NDDSHOME}"/bin/rtiddsgen -language python "$filename" -d ./ -unboundedSupport
done

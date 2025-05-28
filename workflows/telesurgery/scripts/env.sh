#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/" >/dev/null 2>&1 && pwd)"

# RTI Home
export NDDSHOME=/opt/rti.com/`ls /opt/rti.com/ | tail -n1`

# RTI QOS Profile
export NDDS_QOS_PROFILES=$SCRIPT_DIR/dds/qos_profile.xml

# RTI Discovery Address
export NDDS_DISCOVERY_PEERS=10.111.66.170

# RTI License
if [ -z "${RTI_LICENSE_FILE}" ]; then
  export RTI_LICENSE_FILE=$SCRIPT_DIR/dds/rti_license.dat
fi

# Python Path
export PYTHONPATH=$SCRIPT_DIR

# Optional: NTP Server to capture time diff between 2 nodes
export NTP_SERVER_HOST=pool.ntp.org
export NTP_SERVER_PORT=123

#!/bin/bash

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

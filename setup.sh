#! /bin/bash

PROTOC_URL=https://github.com/protocolbuffers/protobuf/releases/download/v3.19.6/protoc-3.19.6-linux-x86_64.zip
PROTOC=protoc/bin/protoc

function install_protoc() (
  if [ ! -x "$PROTOC" ]; then
    echo "Install protoc."
    mkdir protoc
    cd protoc
    curl -L -o protoc.zip "$PROTOC_URL"
    unzip protoc.zip
  else
    echo "Found protoc."
  fi
)

function run_protoc() (
  PROTO_NAMES=$(find swirl_lm -name '*.proto')
  for proto in ${PROTO_NAMES}; do
    "$PROTOC" -I=. --python_out=. $proto
  done
)

function install_swirl_lm() {
  python3 -m pip install --force-reinstall .
}

cd $(dirname "$0")
install_protoc
run_protoc
install_swirl_lm

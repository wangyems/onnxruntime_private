#!/bin/bash
set -e -x

PYTHON_EXE=$1
${PYTHON_EXE} -m pip install decorator scipy

apt-get update && apt-get install -y libtinfo5

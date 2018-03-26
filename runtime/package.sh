#!/bin/bash

set -x

export PKG_NAME=kaena-runtime
export PKG_ARCH=`uname -m`

# RPMDIR -must- be set
if [ -z "${RPMDIR}" ]; then
	echo "RPMDIR is not set"
	exit 1
fi
export RPMDIR

source ${BUILDFRAMEWORKDIR:-../aes-buildframework}/functions/package.sh

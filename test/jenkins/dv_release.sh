#!/bin/bash
set -e

RELEASE_ROOT=/proj/trench/sw/kaena-release
cp krt-*.*-dv-hal.tar.gz ${RELEASE_ROOT}

cd ${RELEASE_ROOT}
RELEASE_NAME=`ls *.tar.gz | sort -V -r | head -n 1 | xargs -I{} basename {} .tar.gz`
RELEASE_DIRECTORY=${RELEASE_ROOT}/${RELEASE_NAME}
echo 'Release directory ${RELEASE_DIRECTORY}'

# Create release directory and unzip tar into it.
mkdir -p ${RELEASE_DIRECTORY}
tar -xzf ${RELEASE_NAME}.tar.gz -C ${RELEASE_DIRECTORY}
unlink latest
ln -s ${RELEASE_DIRECTORY} latest

# Fix up permissions so everyone in the 'ml' group can read.
chgrp -R ml ${RELEASE_DIRECTORY}
chmod -R g+r ${RELEASE_DIRECTORY}

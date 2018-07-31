#!/bin/bash
set -e

RELEASE_ROOT=/proj/trench/sw/kaena-release
cd ${RELEASE_ROOT}
RELEASE_NAME=`find -maxdepth 1 -type f -name "*.tar.gz" -printf "%T+ %p\n" | sort -r | head -n 1 | cut -d " " -f 2 | xargs -I{} basename {} .tar.gz`
RELEASE_DIRECTORY=${RELEASE_ROOT}/${RELEASE_NAME}

# Create release directory and unzip tar into it.
mkdir ${RELEASE_DIRECTORY}
tar -xzf ${RELEASE_NAME}.tar.gz -C ${RELEASE_DIRECTORY}
ln -s ${RELEASE_DIRECTORY} latest

# Fix up permissions so everyone in the 'ml' group can read/write(/delete).
chgrp -R ml ${RELEASE_DIRECTORY}
chmod -R g+rw ${RELEASE_DIRECTORY}

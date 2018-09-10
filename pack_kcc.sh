#!/bin/bash

if [ "$KAENA_PATH" = "" ] || [ "$BLD_DIR" = "" ]; then
    echo "Enviorment varaibles not set..."
    echo "Exiting" && exit 1
fi

KCC_INSTALL_DIR=${BLD_DIR}/tmp_install_kcc

# get version and patch
cd $KAENA_PATH || exit 1
version=$(cat $KAENA_PATH/version 2> /dev/null || (echo 0.1.0 > $KAENA_PATH/version && cat $KAENA_PATH/version))
patch=$(git rev-list $(git rev-list --no-walk --max-count=1 --tags 2> /dev/null)..HEAD --count 2> /dev/null)

# create installation files and directory
rm -rf $KCC_INSTALL_DIR
mkdir -p $KCC_INSTALL_DIR || exit 1
(cd $KAENA_PATH && make install PREFIX=$KCC_INSTALL_DIR) || exit 1

pack_dir=$1
if [ $# -eq 0 ]; then
   pack_dir=${KCC_BLD_DIR}
fi

# pack installation files into one file
mkdir -p ${pack_dir} || exit 1
cd ${KCC_INSTALL_DIR}
tar -pcvzf ${pack_dir}/kcc-${version}-${patch}.tar.gz bin lib

# clean the created directories
rm -rf $KCC_INSTALL_DIR

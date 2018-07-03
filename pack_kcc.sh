#!/bin/bash

SOURCE_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

(source $SOURCE_DIR/../shared/set_env.sh) || (>&2 echo "'shared/set_env.sh' does not exist or script in incorrect directory" && exit 1)

# get version and patch
cd $KAENA_PATH || exit 1
version=$(cat $KAENA_PATH/version 2> /dev/null || (echo 0.1.0 > $KAENA_PATH/version && cat $KAENA_PATH/version))
patch=$(git rev-list $(git rev-list --no-walk --max-count=1 --tags 2> /dev/null)..HEAD --count 2> /dev/null)

# create installation files and directory
rm -rf $KCC_INSTALL_DIR
mkdir -p $KCC_INSTALL_DIR || exit 1
(cd $KAENA_PATH && make install PREFIX=$KCC_INSTALL_DIR) || exit 1


# pack installation files into one file
mkdir -p ${KCC_BLD_DIR} || exit 0
cd ${KCC_INSTALL_DIR}
tar -pcvzf ${KCC_BLD_DIR}/kcc-${version}-${patch}.tar.gz bin lib

# clean the created directories
rm -r $KCC_INSTALL_DIR

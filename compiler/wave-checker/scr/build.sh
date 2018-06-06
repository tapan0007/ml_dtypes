#!/bin/sh
WC_PATH=${KAENA_PATH}/compiler/wave-checker
WC_BUILD_DIR=wc-build
mkdir ${WC_BUILD_DIR} || exit 1
cd ${WC_BUILD_DIR} || exit 1
${WC_PATH}/scr/gtest-build.sh || exit 1
cmake ${WC_PATH} -DGTEST_DIR=`realpath ./gtest-build-dir` || exit 1
make || exit 1

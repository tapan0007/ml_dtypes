#!/bin/sh
WC_PATH=${KAENA_PATH}/compiler/wave-checker
WC_BUILD_DIR=wc-build
#if test $1 -eq clean_command; then
if [ "$1" = "clean" ]; then
  if test -d ${WC_BUILD_DIR}; then
    \rm -rf ${WC_BUILD_DIR} || exit 1
    echo "INFO::${WC_BUILD_DIR} directory is completely deleted" || exit 1
  fi
  if test -f ${WC_PATH}/CMakeCache.txt; then
    \rm ${WC_PATH}/CMakeCache.txt || exit 1
    echo "INFO::${WC_PATH}/CMakeCache.txt is deleted" || exit 1
  fi
  if test -d ${WC_PATH}/CMakeFiles; then
    \rm -rf ${WC_PATH}/CMakeFiles || exit 1
    echo "INFO::${WC_PATH}/CMakeFiles directory is completely deleted" || exit 1
  fi
else
  mkdir ${WC_BUILD_DIR} || exit 1
  cd ${WC_BUILD_DIR} || exit 1
  ${WC_PATH}/scr/gtest-build.sh || exit 1
  cmake ${WC_PATH} -DGTEST_DIR=`realpath ./gtest-build-dir` || exit 1
  make || exit 1
fi

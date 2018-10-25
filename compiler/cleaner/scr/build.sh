#!/bin/sh
TOOL_PATH=$2
TOOL_BUILD_DIR=$3
#if test $1 -eq clean_command; then
if [ "$1" = "clean" ]; then
  if test -d ${TOOL_BUILD_DIR}; then
    \rm -rf ${TOOL_BUILD_DIR} || exit 1
    echo "INFO::${TOOL_BUILD_DIR} directory is completely deleted" || exit 1
  fi
  if test -f ${TOOL_PATH}/CMakeCache.txt; then
    \rm ${TOOL_PATH}/CMakeCache.txt || exit 1
    echo "INFO::${TOOL_PATH}/CMakeCache.txt is deleted" || exit 1
  fi
  if test -d ${TOOL_PATH}/CMakeFiles; then
    \rm -rf ${TOOL_PATH}/CMakeFiles || exit 1
    echo "INFO::${TOOL_PATH}/CMakeFiles directory is completely deleted" || exit 1
  fi
else
  mkdir ${TOOL_BUILD_DIR} || exit 1
  cd ${TOOL_BUILD_DIR} || exit 1
  cmake ${TOOL_PATH} || exit 1
  make || exit 1
fi

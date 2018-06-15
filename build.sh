#!/bin/sh

cd $KAENA_PATH && make "$@" || exit 1
cd $KAENA_PATH/compiler/wave-checker && ./scr/build.sh clean && ./scr/build.sh || exit 1

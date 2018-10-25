#!/bin/sh

cd $KAENA_PATH && make "$@" || exit 1
cd $KAENA_PATH/compiler/wave-checker && ./scr/build.sh clean && ./scr/build.sh || exit 1
cd $KAENA_PATH/compiler/cleaner && ./scr/build.sh clean $KAENA_PATH/compiler/cleaner $KAENA_PATH/compiler/cleaner/build && ./scr/build.sh noclean $KAENA_PATH/compiler/cleaner $KAENA_PATH/compiler/cleaner/build || exit 1

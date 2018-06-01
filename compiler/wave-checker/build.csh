#!/bin/csh -f
set WC_PATH=$KAENA_PATH/compiler/wave-checker
mkdir wc-build
cd wc-build
source $WC_PATH/test/gtest-build
cmake $WC_PATH
make

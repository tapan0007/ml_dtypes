#!/bin/sh
git clone https://github.com/google/googletest.git gtest || exit 1
mkdir gtest-build-dir || exit 1
cd gtest-build-dir || exit 1
cmake -DCMAKE_INSTALL_PREFIX:PATH=`realpath .` -DCMAKE_CXX_FLAGS:STRINGS=-std=c++11 ../gtest || exit 1
make || exit 1
make install || exit 1
cd .. || exit 1

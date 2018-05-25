git clone https://github.com/google/googletest.git gtest
mkdir gtest-build-dir;cd gtest-build-dir
export GTEST_DIR=`realpath .`
cmake -DCMAKE_INSTALL_PREFIX:PATH=. ../gtest
make;make install
cd ..

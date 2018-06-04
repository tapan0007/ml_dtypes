WC_PATH=${KAENA_PATH}/compiler/wave-checker
mkdir wc-build
cd wc-build
source ${WC_PATH}/scr/gtest-build.csh
cmake ${WC_PATH}
make

#!/bin/csh -f

#git bisect start
#git bisect good 1d145c6bb4f0fec0328b134b534d28709872e7d4
#git bisect bad 82a81c3f177ee0f95b107e4292e8e5ba3617a1b3
#git bisect run /home/zejdaj/bin/git_bisect_kaena.csh --test 0-2matmult_add_fp32_wave | & tee log

set d = `pwd`

cd $KAENA_PATH/compiler/be
#git pull
make clean
make
cd $KAENA_PATH
make

cd $d
$KAENA_PATH/test/e2e/RunAll $*

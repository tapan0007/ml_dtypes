#/bin/csh -f

# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
# Validation of the npy_conv2d multi-type golden calculation against Tensorflow
# Kaena core and backend (scheduler, tcc, incling) are not involved

ln -s trivnet_input:0_NCHW.npy i
ln -s trivnet_1conv__weight1__read:0_MCRS.npy w
ln -s trivnet_1conv__i1:0_NCHW.npy otf

# strided

set c = 32; set o = 16
set s = 2
set strides = "1 1 $s $s"

set log = log-conv2d

foreach p ( b1-h4-r2-s$s-c1-m1-wmin1-wmax4-imin5-imax20  b2-h16-r3-s$s-c1-m1-wmin-0.1-wmax0.1-imin5-imax20  )

  make -f $KAENA_PATH/compiler/tffe/test/Makefile trivnet_conv1 NN_CONFIG=$p OUT_PREFIX=trivnet_ NN_NAME=1conv

  $KAENA_PATH/compiler/util/npy_conv2d  --image i NCHW  --weight w MCRS  --output o$c-$o.npy  NCHW  --floats float$c  float$o --strides $strides | tee log-$c-$o

  $KAENA_PATH/compiler/util/npy_diff_files --verbose 3 --gold o$c-$o.npy --new otf | tee -a $log
end

# non-strided

set c = 64; set o = 64
set strides = "1 1 1 1"

foreach p ( b2-h4-r3-s1-c5-m6-wmin-0.1-wmax0.1-imin-1-imax1 )
  make -f $KAENA_PATH/compiler/tffe/test/Makefile trivnet_conv1 NN_CONFIG=$p OUT_PREFIX=trivnet_ NN_NAME=1conv

$KAENA_PATH/compiler/util/npy_conv2d  --image i NCHW  --weight w MCRS  --output o$c-$o.npy  NCHW  --floats float$c  float$o --strides $strides | tee log-$c-$o

  $KAENA_PATH/compiler/util/npy_diff_files --verbose 3 --gold o$c-$o.npy --new otf | tee -a $log

end

echo
echo -----------------------------------------
echo
egrep -A 1 "Relative Error" $log

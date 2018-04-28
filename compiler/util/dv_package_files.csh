#!/bin/csh -f

#cd /work1/zejdaj/r4/dv_tests

cat > README.txt <<EOF
0-1conv0_wave  Simplest single convolution pixel 2 * weight 3 = pixel 6

0-1conv0m8_wave  As above but 8 channels

0-act_wave  2x2 image, 128 channels tanh

0-1conv1ba1_h55c64m64_wave  55x55x filter 1x1  64 channel convolution

3-rn50_pool2_wave   Average pool 7x7 from resnet 50

EOF


( setenv SIM_ADD_FLAGS '--debugflags tpb_exec ' ; $KAENA_PATH/test/e2e/RunAll --test 0-1conv0_wave  0-1conv0m8_wave 0-act_wave 0-1conv1ba1_h55c64m64_wave  3-rn50_pool2_wave  ) |& tee log

foreach d ( `/bin/ls -d [0-9]-*` )
  cd $d
  
  # Visualization
  $KAENA_PATH/compiler/util/kelf2dot --json nn_graph.json
  mv nn_graph-subgraphs.dot.svg nn_kelf.svg
  
  $KAENA_PATH/compiler/util/kgraph2dot --json sg00/wavegraph-out.json
  cp sg00/wavegraph-out-layers.dot.svg kgraph.svg
  cp sg00/wavegraph-out-waveops.dot.svg wavegraph.svg
  
  cp trivnet_graph.dot.svg dataflow.svg
  cp trivnet_ops.csv dataflow_ops.csv
  
  # Profiles
  $KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-*.txt --tpb sg00/*.tpb --verbose;  mv out_profile.svg tpb_profile.svg
  
  # package
  set p = ../package/$d
  mkdir -p $p
  cp *.svg  $p
  /bin/rm $p/trivnet*
  cp sg00/TrivNet-*.tpb $p
  rename s/TrivNet-/program-/ $p/*.tpb
  set npy_files = `grep npy sg00/wavegraph-out.json | perl -pe 's/.*: "|",?//g'`
  echo '' >! npy_list.txt
  foreach f ($npy_files)
    find -name $f | head -1 >> npy_list.txt
  end
  tar cvz -f $p/npy_files -T npy_list.txt
  
  cd ..
end


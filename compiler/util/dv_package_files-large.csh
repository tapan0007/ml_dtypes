#!/bin/csh -f

#cd /work1/zejdaj/r4/dv_tests_large

cat > README.txt <<EOF
7-rn50_nne_fp16_wave    Full Resnet50 fp16 inference all layers on Inkling except the softmax

EOF


#( setenv SIM_ADD_FLAGS '--debugflags tpb_exec --dram_latency 0 --dram_frequency 0' ; $KAENA_PATH/test/e2e/RunAll --test 7-rn50_nne_fp16_wave  ) |& tee log

foreach d ( `/bin/ls -d [0-9]-*` )
  cd $d
  
  # Visualization
  $KAENA_PATH/compiler/util/kelf2dot --json nn_graph.json
  mv nn_graph-subgraphs.dot.svg nn_kelf.svg
  
  $KAENA_PATH/compiler/util/kgraph2dot --json sg00/wavegraph-out.json --waveop_range 0 3000
  cp sg00/wavegraph-out-layers.dot.svg kgraph.svg
  cp sg00/wavegraph-out-waveops.dot.svg wavegraph.svg
  
  cp trivnet_graph.dot.svg dataflow.svg
  cp trivnet_ops.csv dataflow_ops.csv
  
  # Profiles
  #$KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-*.txt --tpb sg00/*.tpb --long 2e5 ;  mv out_profile.png tpb_profile.png
  
  # package
  set p = ../package/$d
  mkdir -p $p
  cp *.svg  $p
  cp tpb_profile.png  $p
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


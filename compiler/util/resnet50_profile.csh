#!/bin/csh -f

if ! ( -d "$KAENA_PATH" && -d "$INKLING_PATH" && -d "$KAENA_EXT_PATH" && -d "$ARCH_ISA_PATH" ) then
  echo ERROR: Setup Kaena envvar and restart
  exit 1
endif

# Compile inkling with profiling enabled
  pushd $INKLING_PATH/sim; make clean; make opt
  popd
#exit
# Run the test with profile info from Inkling
mkdir default
pushd default
( setenv SIM_ADD_FLAGS '--debug_flags tpb_exec ' ; $KAENA_PATH/test/e2e/RunAll --test 7-rn50_nne_fp16_wave ) >& log ;  $KAENA_PATH/compiler/util/tpb_profile --log 7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt  --tpb 7-rn50_nne_fp16_wave/sg00/*.tpb --long 2e6 --show >& log1; mv out_profile.png ../tpb_profile_rn50_float16_default.png; $KAENA_PATH/compiler/util/tpb_profile --log 7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt --tpb 7-rn50_nne_fp16_wave/sg00/*.tpb --event_statistics > event_stats.txt &
popd

mkdir fast_dram
pushd fast_dram
( setenv SIM_ADD_FLAGS '--debug_flags tpb_exec --dram_latency 0 --dram_frequency 0' ; $KAENA_PATH/test/e2e/RunAll --test 7-rn50_nne_fp16_wave ) >& log;  $KAENA_PATH/compiler/util/tpb_profile --log 7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt --tpb 7-rn50_nne_fp16_wave/sg00/*.tpb --long 2e5 --show >& log1; mv out_profile.png ../tpb_profile_rn50_float16_fast_dram.png &
popd

mkdir two_banks
pushd two_banks
( setenv SIM_ADD_FLAGS '--debug_flags tpb_exec --dram_frequency 6400' ; $KAENA_PATH/test/e2e/RunAll --test 7-rn50_nne_fp16_wave ) >& log ;  $KAENA_PATH/compiler/util/tpb_profile --log 7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt  --tpb 7-rn50_nne_fp16_wave/sg00/*.tpb --long 2e6 --show >& log1; mv out_profile.png ../tpb_profile_rn50_float16_two_banks.png &
popd

wait

# Open detailed interactive profile debug
echo 'Run interactive by   $KAENA_PATH/compiler/util/tpb_profile --log fast_dram/7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt --tpb fast_dram/7-rn50_nne_fp16_wave/sg00/*.tpb --verbose --show --cycle_range 1e6 1.1e6'

# Open full-network hierachical fast interactive profile debug
echo 'Run interactive by   $KAENA_PATH/compiler/util/tpb_profile --log default/7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt --tpb default/7-rn50_nne_fp16_wave/sg00/*.tpb --show --cycle_range 0 1e9 --resolution 300'

# Event statistics
echo '$KAENA_PATH/compiler/util/tpb_profile --log default/7-rn50_nne_fp16_wave/working_dir/log-exec-sg00-wave.txt --tpb default/7-rn50_nne_fp16_wave/sg00/*.tpb --event_statistics > event_stats.txt'

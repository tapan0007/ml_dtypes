#!/bin/csh -f

if ! ( -d "$KAENA_PATH" && -d "$INKLING_PATH" && -d "$KAENA_EXT_PATH" && -d "$ARCH_ISA_PATH" ) then
  echo ERROR: Setup Kaena envvar and restart
  exit 1
endif

# Add --debug_flags tpb_exec to sim
#grep tpb_exec $KAENA_PATH/runtime/util/runtime_sim || exit 1


# Compile inkling with profiling enabled
  pushd $INKLING_PATH/sim; make clean; make -j 16 opt
  popd
#exit
# Run the test with profile info from Inkling
set tests = ( \
  7-rn50_nne_fp16_wave \
  7-rn50_nne_fp16_wave-fast_dram \
  7-rn50_nne_fp16_wave-two_banks \
  8-rn50_nne_fp16_b16_wave \
  8-rn50_nne_fp16_b16_wave-fast_dram \
  8-rn50_nne_fp16_b16_wave-two_banks \
  )

#$KAENA_PATH/test/e2e/RunAll --test $tests


foreach t ($tests)
  pushd $t
    $KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-wave.txt  --tpb sg00/*.tpb --long 2e6  --show >& log1; \
    mv out_profile.png ../tpb_profile_$t.png; \
    $KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-wave.txt --tpb sg00/*.tpb --event_statistics > event_stats.txt &
  popd
end

wait

# Open detailed interactive profile debug
echo 'Run interactive by   $KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-wave.txt --tpb sg00/*.tpb --verbose --show --cycle_range 1e6 1.1e6'

# Open full-network hierachical fast interactive profile debug
echo 'Run interactive by   $KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-wave.txt --tpb sg00/*.tpb --show --cycle_range 0 1e9 --resolution 300'

# Event statistics
echo '$KAENA_PATH/compiler/util/tpb_profile --log working_dir/log-exec-sg00-wave.txt --tpb sg00/*.tpb --event_statistics > event_stats.txt'

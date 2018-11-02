# Copyright (C) 2017, Amazon.com. All Rights Reserved

# Kaena compiler TF front end and flow regression harness
# See RunAll for top-level user interface 
#
# Examples
#   cd /tmp (some empty dir)
#   setenv KAENA_PATH /your/install/path/of/Kaena
#   make -f $KAENA_PATH/compiler/tffe/Makefile  jdr_v2  NN_CONFIG=b1-h4-r3-s1-c2-m1-wmin-0.1-wmax0.2-imin1-imax5  OUT_PREFIX=trivnet_  NN_NAME=1conv
#     # For NN_CONFIG semantic see the NN generator  compiler/util/tffe_sweep_1conv
#   This generates data flow graph svg, K-graph package tgz

# TO_DO add test env using https://docs.python.org/2/library/unittest.html

#on MAC OSX 10.12.6
#need pip3 install graphviz
#need PIL:
#pip3 install pillow
#need graphviz, so use
#brew install graphviz
#if need brew:  /usr/bin/ruby -e "$(curl -fsSL https://raw.githubusercontent.com/Homebrew/install/master/install)"

# On ubuntu ML AMI use pip from AMI TF directory (DO NOT USE pip3 or detault pip):
#   pip install graphviz pillow

MKFILE := $(abspath $(lastword $(MAKEFILE_LIST)))
#KAENA_PATH = ../..
OUT_PREFIX = "out_"
NN_NAME = "jdr"
DATA_TYPE = "float16"
NN_GRAPH_FILE = "nn_graph.json"
TF_PATH = $(shell python3 -c "import tensorflow as tf; print(tf.__file__.replace('/__init__.py', ''))")
RT_ARGS = ""

export KCC := $(KAENA_PATH)/compiler
export KAENA_E2E := $(KAENA_PATH)/test/e2e
export KAENA_EXT := $(KAENA_EXT_PATH)
export PYTHONPATH := $(KCC)/tffe:$(PYTHONPATH)
$(info KAENA_PATH is ${KAENA_PATH})
$(info KCC is ${KCC})
$(info TF_PATH is ${TF_PATH})
$(info DATA_TYPE is ${DATA_TYPE})

KCC_COMMAND = $(KCC)/scripts/tffe --out_prefix $(OUT_PREFIX) --verbose 1 --tfpb $(PB_FILE) --depth 5 --width 30 $(NN_ARGS)
ifeq ($(SEALIFE), 1)
  KCC_COMMAND = $(AWS_TONGA_SRC)/starfish/bin/kcc_tf --verbose 1 --tfpb $(PB_FILE) $(NN_ARGS)
endif

# Internal targets
compile_pb:
	$(KCC_COMMAND) > log-fe.txt 2>&1

run_kelf:
	ln -s $(PB_FILE) cached_tf.pb
	$(KAENA_PATH)/runtime/util/nn_executor --kelf_dir . --tfpb $(PB_FILE) --check_against_ref all $(RT_ARGS) > log-rt.txt 2>&1


# Main test flow targets

## --debug 3

trivnet_%:
	python3 ${KAENA_E2E}/$@.py $(NN_CONFIG) $(OUT_PREFIX) $(NN_NAME) $(DATA_TYPE)
	export CUDA_VISIBLE_DEVICES=""; python3 $(TF_PATH)/python/tools/freeze_graph.py --input_graph $(OUT_PREFIX)graph.pb --input_checkpoint $(OUT_PREFIX)checkpoint.data --output_graph $(OUT_PREFIX)freeze.pb --output_node_names $(NN_NAME)/output
	$(MAKE) -f $(MKFILE) compile_pb  PB_FILE=$(OUT_PREFIX)freeze.pb
	$(MAKE) -f $(MKFILE) run_kelf    PB_FILE=$(OUT_PREFIX)freeze.pb

tf_pb:
	$(MAKE) -f $(MKFILE) compile_pb  PB_FILE=$(KAENA_EXT)/apps/tf/$(NN_CONFIG)
	$(MAKE) -f $(MKFILE) run_kelf    PB_FILE=$(KAENA_EXT)/apps/tf/$(NN_CONFIG)

tf_s3:
	aws --profile=kaena s3 ls $(NN_CONFIG)/$(NN_NAME)
	aws --profile=kaena s3 cp $(NN_CONFIG)/$(NN_NAME) . --only-show-errors
	$(MAKE) -f $(MKFILE) compile_pb  PB_FILE=$(NN_NAME)
	$(MAKE) -f $(MKFILE) run_kelf    PB_FILE=$(NN_NAME)
	rm $(NN_NAME) # Free up some space

ckpt_%:
	python3 ${KAENA_E2E}/$@.py $(NN_CONFIG) $(OUT_PREFIX) $(NN_NAME) $(DATA_TYPE)
	$(MAKE) -f $(MKFILE) compile_pb  PB_FILE=$(OUT_PREFIX)model
	$(KAENA_PATH)/runtime/util/nn_executor --kelf_dir . --tfpb $(OUT_PREFIX)model --check_against_ref all $(RT_ARGS) > log-rt.txt 2>&1

run_kelf_cached:
	rsync -av --exclude log-\*.txt $(CACHED_KELF)/* . || aws --profile=kaena s3 cp $(CACHED_KELF)/"*" . --recursive --exclude "log-*" --quiet
	/bin/rm -rf */working_dir
	$(KAENA_PATH)/runtime/util/nn_executor --kelf_dir . --tfpb cached_tf.pb --check_against_ref all $(RT_ARGS) > log-rt.txt 2>&1


# Tests that run as commit criteria.
# The set after the blank line are auto-derived tests created by appending _qemu suffix to an explicit test.
TEST_ARGS_CHECK = --test 0-1conv0_wave  3-rn50-t00_wave 3-rn50_pool2_wave 4-ptb_word_small1_host 3-rn50-13_wave 0-1conv1ba1_h4c2m2_wave 0-1conv0_qemu_wave 0-1conv0_b16_wave 0-1conv0_ckpt_wave 0-1conv_h4r1_b2_wave 0-1conv_h4r1_wave 0-1conv_tile_r1_wave 0-1conv_tile_r1h32_wave 0-4conv_relu_nne 0-116conv_tanh_wave 0-4conv_multiout_wave 5-rn50_nne_to_act4_wave-no_repl-t1_focus_to-qemu-all_layers 5-parwavenet_10_fp16_in_to_add1_wave \
\
0-11conv_tanh_wave_qemu  0-1clipbyvalue_wave_qemu  0-1concat_h1c1m1ni2_qemu  0-1conv0_subnormal_wave_qemu  \
0-1conv1ba1_h4c2m2_wave_qemu  0-1conv_exp_pad_wave_qemu  0-1conv_h4_softplus_wave_qemu  \
4-ptb_word_small_sigmoid_wave 4-ptb_word_small_sigmoid_2l_wave_qemu \
4-ptb_word_small_sigmoid_fp16_b32_wave_qemu \
0-resadd_wave_qemu   

# Tests that will be run on emulator. These tests are run to check if RTL is fine.
# Using parallel 1 till we figure out the lock issue
TEST_ARGS_CHECK_EMU= --force_qemu --parallel 1 --test \
5-inceptionv3_wave_dog_sg00_tpb_upto_concat1 \
5-inceptionv3_wave_dog_sg00_tpb_upto_concat2 \
6-inceptionv3_wave_dog_sg00_tpb_upto_concat4 \
6-inceptionv3_wave_dog_sg00_tpb_upto_concat5 \
6-inceptionv3_wave_dog_sg00_tpb_upto_concat8 \
6-inceptionv3_wave_dog_sg00_tpb_upto_concat9 \
7-inceptionv3_wave_dog_sg00_tpb_upto_concat10 \
7-inceptionv3_wave_dog_sg00_tpb_upto_concat11 \
4-ptb_word_small_sigmoid_fp16_b32_wave \
5-rn50_nne_to_act13_wave-no_repl \
5-rn50_nne_to_act13_wave-repl \
5-rn50_nne_to_act4_wave-no_repl \
5-rn50_nne_to_act4_wave-repl \
6-rn50_nne_to_act22_wave-no_repl \
6-rn50_nne_to_act22_wave-repl \
6-rn50_nne_to_act25_wave-repl \
6-rn50_nne_to_act28_wave-repl \
6-rn50_nne_to_act37_wave-repl \
6-rn50_nne_to_act40_wave-repl \
6-rn50_nne_to_act43_wave-repl \
6-rn50_nne_to_act46_wave-no_repl \
6-rn50_nne_to_act46_wave-repl \
6-rn50_nne_to_act49_wave-no_repl \
6-rn50_nne_to_act49_wave-repl \
7-rn50_nne_fp16_b4_wave \
7-rn50_nne_fp16_b4_wave-no_repl \
7-rn50_nne_fp16_wave-no_repl \
6-parwavenet_10_fp16_tanh_to_add9_wave \
6-parwavenet_10_fp16_tanh_to_add12_wave \
6-parwavenet_10_fp16_tanh_to_add15_wave \
6-parwavenet_10_fp16_tanh_to_add18_wave \
6-parwavenet_10_fp16_tanh_to_add22_wave \
7-parwavenet_10_fp16_tanh_to_add24_wave \
7-amoebanet_fp16_cell1 \
7-amoebanet_fp16_cell11 \
7-amoebanet_fp16_cell5 \
7-amoebanet_fp16_cell9

# 0-subtract_psum_sb_wave  

# Tests that we want to run on qemu but they still fails as of Oct 2018
# 0-1conv_h4_sigmoid_wave_qemu  0-1stridedslice_tanh_sigmoid_wave_qemu  0-scaleadd_wave_qemu

TEST_ARGS_CHECK_LEVEL7 = --level 7 --select not_waived

check:
	${KAENA_PATH}/test/e2e/RunAll ${TEST_ARGS_CHECK}

check_level7:
	${KAENA_PATH}/test/e2e/RunAll ${TEST_ARGS_CHECK_LEVEL7}

check_emu:
	${KAENA_PATH}/test/e2e/RunAll ${TEST_ARGS_CHECK_EMU}

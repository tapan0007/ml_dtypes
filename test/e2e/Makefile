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

trivnet_%:
	python3 ${KAENA_E2E}/$@.py $(NN_CONFIG) $(OUT_PREFIX) $(NN_NAME) $(DATA_TYPE)
	export CUDA_VISIBLE_DEVICES=""; python3 $(TF_PATH)/python/tools/freeze_graph.py --input_graph $(OUT_PREFIX)graph.pb --input_checkpoint $(OUT_PREFIX)checkpoint.data --output_graph $(OUT_PREFIX)freeze.pb --output_node_names $(NN_NAME)/output
	$(KCC)/scripts/tffe --out_prefix $(OUT_PREFIX) --verbose 1 --tfpb $(OUT_PREFIX)freeze.pb --depth 5 --width 30 $(NN_ARGS) > log-fe.txt 2>&1
	$(KAENA_PATH)/runtime/util/nn_executor --kelf_dir . --tfpb $(OUT_PREFIX)freeze.pb --check_against_ref $(RT_ARGS) > log-rt.txt 2>&1
tf_pb:
	$(KCC)/scripts/tffe --out_prefix $(OUT_PREFIX) --verbose 1 --tfpb $(KAENA_EXT)/apps/tf/$(NN_CONFIG) --depth 5 --width 30 $(NN_ARGS) > log-fe.txt 2>&1
	$(KAENA_PATH)/runtime/util/nn_executor --kelf_dir . --tfpb $(KAENA_EXT)/apps/tf/$(NN_CONFIG) --check_against_ref $(RT_ARGS) > log-rt.txt 2>&1

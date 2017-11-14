# Examples
#   cd /tmp (some empty dir)
#   setenv KAENA_PATH /your/install/path/of/Kaena
#   make -f $KAENA_PATH/compiler/tffe/Makefile  jdr_v2  NN_CONFIG=b1-h4-r2-s1-c2-m1-wmin-0.1-wmax0.2-imin1-imax5  OUT_PREFIX=trivnet_  NN_NAME=1conv
#     # For NN_CONFIG semantic see the NN generator  compiler/util/tffe_sweep_1conv
#   This generates data flow graph svg, K-graph package tgz

# TO_DO add test env using https://docs.python.org/2/library/unittest.html


#KAENA_PATH = ../..
OUT_PREFIX = "out_"
NN_NAME = "jdr"
DATA_TYPE = "float16"
TF_PATH = $(shell python3 -c "import tensorflow as tf; print(tf.__file__.replace('/__init__.py', ''))")
export KAENA_TFFE := $(KAENA_PATH)/compiler/tffe
export PYTHONPATH := $(KAENA_TFFE):$(PYTHONPATH)
$(info KAENA_PATH is ${KAENA_PATH})
$(info KAENA_TFFE is ${KAENA_TFFE})
$(info TF_PATH is ${TF_PATH})
$(info DATA_TYPE is ${DATA_TYPE})

jdr_%:
	python3 $(KAENA_TFFE)/$@.py $(NN_CONFIG) $(OUT_PREFIX) $(NN_NAME) $(DATA_TYPE)
	python3 $(TF_PATH)/python/tools/freeze_graph.py --input_graph $(OUT_PREFIX)graph.pb --input_checkpoint $(OUT_PREFIX)checkpoint.data --output_graph $(OUT_PREFIX)freeze.pb --output_node_names $(NN_NAME)/output
	$(KAENA_TFFE)/tffe.py --out_prefix $(OUT_PREFIX) --verbose 1 --tfpb $(OUT_PREFIX)freeze.pb --depth 5 --images $(OUT_PREFIX)ref_input.npy --width 30
	mv $(OUT_PREFIX)compiler.tgz $(OUT_PREFIX)$(NN_NAME)_$(NN_CONFIG).tgz
        

# Examples
#   cd /tmp (some empty dir)
#   setenv KAENA_PATH /your/install/path/of/Kaena
#   make -f $KAENA_PATH/compiler/tffe/Makefile  jdr_v2  NN_CONFIG=b1-h4-r2-s1-c2-m1-wmin-0.1-wmax0.2-imin1-imax5  OUT_PREFIX=trivnet_  NN_NAME=1conv



#KAENA_PATH = ../..
OUT_PREFIX = "out_jdr_v2"
TF_PATH = $(shell python3 -c "import tensorflow as tf; print(tf.__file__.replace('/__init__.py', ''))")
export KAENA_TFFE := $(KAENA_PATH)/compiler/tffe
export PYTHONPATH := $(KAENA_TFFE):$(PYTHONPATH)
$(info KAENA_PATH is ${KAENA_PATH})
$(info KAENA_TFFE is ${KAENA_TFFE})
$(info TF_PATH is ${TF_PATH})

jdr_v2:
	python3 $(KAENA_TFFE)/$@.py $(NN_CONFIG) $(OUT_PREFIX) $(NN_NAME)
	python $(TF_PATH)/python/tools/freeze_graph.py --input_graph $(OUT_PREFIX)graph.pb --input_checkpoint $(OUT_PREFIX)checkpoint.data --output_graph $(OUT_PREFIX)freeze.pb --output_node_names $(NN_NAME)/output
	$(KAENA_TFFE)/tffe.py --out_prefix $(OUT_PREFIX) --verbose 1 --tfpb $(OUT_PREFIX)freeze.pb --depth 5 --images linear --width 30
	mv $(OUT_PREFIX)compiler.tgz $(OUT_PREFIX)$(NN_NAME)_$(NN_CONFIG).tgz
        
jdr_v3:
	python3 $@.py
	python /home/ubuntu/src/tensorflow_p3/tensorflow/python/tools/freeze_graph.py --input_graph out_$@.pb --input_checkpoint out_$@.data --output_graph out_"$@"_freeze.pb --output_node_names $@/output
	./tffe.py --tfpb out_"$@"_freeze.pb --depth 5 --images linear --width 30
	python3 -c "import numpy as np; x=np.load('out_"$@"__i2:0.npy'); print('Output OFMAP\n%s\n\nOFMAP as uinit16:\n%s\n' % (x, x.view(dtype=np.uint16)))"
	python3 -c "import numpy as np; x=np.load('out_"$@"__i2:0_NCHW.npy'); print('Output OFMAP NCHW\n%s\n\nOFMAP NCHW as uinit16:\n%s\n' % (x, x.view(dtype=np.uint16)))"
	mv out_compiler.py trivnet.py
	ln -s -f out_input:0_NCHW.npy input.npy
	ln -s -f out_"$@"__i2:0_NCHW.npy golden.npy
	tar cvzf trivnet_v3.tgz trivnet.py input.npy golden.npy out_input:0_NCHW.npy out_"$@"__weight1__read:0_MCRS.npy \
          out_"$@"__i1:0_NCHW.npy out_"$@"__weight2__read:0_MCRS.npy out_"$@"__i2:0_NCHW.npy
        

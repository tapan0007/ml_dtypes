jdr_v2:
	python3 $@.py
	python /home/ubuntu/src/tensorflow_p3/tensorflow/python/tools/freeze_graph.py --input_graph out_$@.pb --input_checkpoint out_$@.data --output_graph out_"$@"_freeze.pb --output_node_names $@/output
	./tffe.py --tfpb out_"$@"_freeze.pb --depth 5 --images linear --width 30
	python3 -c "import numpy as np; x=np.load('out_"$@"__i1:0.npy'); print('Output OFMAP\n%s\n\nOFMAP as uinit16:\n%s\n' % (x, x.view(dtype=np.uint16)))"
	python3 -c "import numpy as np; x=np.load('out_"$@"__i1:0_NCHW.npy'); print('Output OFMAP NCHW\n%s\n\nOFMAP NCHW as uinit16:\n%s\n' % (x, x.view(dtype=np.uint16)))"
	mv out_compiler.py trivnet.py
	ln -s out_input:0_NCHW.npy input.npy
	ln -s out_"$@"__i1:0_NCHW.npy golden.npy
	tar cvzf trivnet_v2.tgz trivnet.py input.npy golden.npy out_input:0_NCHW.npy out_"$@"__weight1__read:0_MCRS.npy \
          out_"$@"__i1:0_NCHW.npy
        
jdr_v3:
	python3 $@.py
	python /home/ubuntu/src/tensorflow_p3/tensorflow/python/tools/freeze_graph.py --input_graph out_$@.pb --input_checkpoint out_$@.data --output_graph out_"$@"_freeze.pb --output_node_names $@/output
	./tffe.py --tfpb out_"$@"_freeze.pb --depth 5 --images linear --width 30
	python3 -c "import numpy as np; x=np.load('out_"$@"__i2:0.npy'); print('Output OFMAP\n%s\n\nOFMAP as uinit16:\n%s\n' % (x, x.view(dtype=np.uint16)))"
	python3 -c "import numpy as np; x=np.load('out_"$@"__i2:0_NCHW.npy'); print('Output OFMAP NCHW\n%s\n\nOFMAP NCHW as uinit16:\n%s\n' % (x, x.view(dtype=np.uint16)))"
	mv out_compiler.py trivnet.py
	ln -s out_input:0_NCHW.npy input.npy
	ln -s out_"$@"__i2:0_NCHW.npy golden.npy
	tar cvzf trivnet_v3.tgz trivnet.py input.npy golden.npy out_input:0_NCHW.npy out_"$@"__weight1__read:0_MCRS.npy \
          out_"$@"__i1:0_NCHW.npy out_"$@"__weight2__read:0_MCRS.npy out_"$@"__i2:0_NCHW.npy
        

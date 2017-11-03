jdr_v3:
	net=jdr_v3
	python3 $@.py
	python /home/ubuntu/src/tensorflow_p3/tensorflow/python/tools/freeze_graph.py --input_graph out_$@.pb --input_checkpoint out_$@.data --output_graph out_"$@"_freeze.pb --output_node_names $@/output
	./tffe.py --tfpb out_"$@"_freeze.pb --depth 5 --images linear --width 30
	python3 -c "import numpy as np; x=np.load('out_"$@"__output:0.npy'); print('Output OFMAP\n', x)"
	mv out_compiler.py trivnet.py
	tar cvzf trivnet_v3_net.tgz trivnet.py out_input:0.npy out_jdr_v3__weight1__read:0.npy out_jdr_v3__weight2__read:0.npy out_jdr_v3__i2:0.npy
        

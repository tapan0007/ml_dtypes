# Layer-Optimizer tests

help:
	@echo "\nList of tests:"
	@ls layeropt_tests
	@echo ""

TESTS=$(wildcard layeropt_tests/*)

gen_test:
	#@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; pushd .; cd $(var); if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/tffe/tffe.py --tfpb graph.pb --image linspace1; rm -f *.npy; tar -xzf *.tgz; else rm -rf $(notdir $(var)); $(KAENA_PATH)/compiler/tffe/test/RunAll --test $(notdir $(var)); cp $(notdir $(var))/*.tgz .; tar -xzf *.tgz; rm -rf $(notdir $(var)); fi; popd;)
	@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; pushd .; cd $(var); if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/tffe/tffe.py --tfpb graph.pb --image linspace1; rm -f *.npy; tar -xzf *.tgz; fi; popd;)

layeropt_sanity:
	cd layeropt_tests/0-2conv3_relu/ && python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json

all:
	@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; make $(notdir $(var)))

%: 
	#cd layeropt_tests/$* && if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/tffe/tffe.py --tfpb graph.pb --image linspace1; rm -f *.npy; tar -xzf *.tgz; python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph out_compiler.json --wavegraph out_wavegraph.json --dot out_wavegraph.svg; fi
	cd layeropt_tests/$* && if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph out_compiler.json --wavegraph out_wavegraph.json --dot out_wavegraph.svg; fi
	cd layeropt_tests/$* && if [ -e trivnet_compiler.json ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json --wavegraph trivnet_wavegraph.json --dot trivnet_wavegraph.svg; fi


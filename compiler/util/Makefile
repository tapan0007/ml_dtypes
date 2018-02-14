# Layer-Optimizer tests

help:
	@echo "\nList of tests:"
	@ls layeropt_tests
	@echo ""

TESTS=$(wildcard layeropt_tests/*)

SANITY_TESTS=0-2conv3_relu 0-1conv_tile 0-1conv1pool resnet50_conv1 resnet50_conv1_biasadd_relu

gen_test:
	#@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; pushd .; cd $(var); if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/tffe/tffe.py --tfpb graph.pb --image linspace1; rm -f *.npy; tar -xzf *.tgz; else rm -rf $(notdir $(var)); $(KAENA_PATH)/compiler/tffe/test/RunAll --test $(notdir $(var)); cp $(notdir $(var))/*.tgz .; tar -xzf *.tgz; rm -rf $(notdir $(var)); fi; popd;)
	@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; pushd .; cd $(var); if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/tffe/tffe.py --tfpb graph.pb --image linspace1; rm -f *.npy; tar -xzf *.tgz; fi; popd;)

layeropt_sanity:
	cd layeropt_tests/0-2conv3_relu/ && python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json 

all:
	@$(foreach var, $(TESTS), make $(notdir $(var));)

sanity:
	@$(foreach var,$(SANITY_TESTS),make $(var);)

%: 
	@echo "******************* $* (layeropt_tests/$*/$*.log) ********************" && cd layeropt_tests/$* && if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph out_compiler.json --wavegraph out_wavegraph.json --dot out_wavegraph.svg > $*.log && grep -e ERROR -e PASS -e FAIL $*.log; fi && if [ -e trivnet_compiler.json ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json --wavegraph trivnet_wavegraph.json --dot trivnet_wavegraph.svg > $*.log && grep -e ERROR -e PASS -e FAIL $*.log; fi



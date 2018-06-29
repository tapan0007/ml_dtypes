# Layer-Optimizer tests

help:
	@echo "\nList of tests:"
	@ls tests
	@echo ""

TESTS=$(wildcard tests/*)

SANITY_TESTS=0-2conv3_relu 0-1conv_tile 0-1conv1pool 0-1conv1maxpool_k3d2

gen_test:
	#@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; pushd .; cd $(var); if [ -e graph.pb ]; then python3 $(KAENA_PATH)/compiler/tffe/tffe.py --tfpb graph.pb --image linspace1; rm -f *.npy; tar -xzf *.tgz; else rm -rf $(notdir $(var)); $(KAENA_PATH)/compiler/tffe/test/RunAll --test $(notdir $(var)); cp $(notdir $(var))/*.tgz .; tar -xzf *.tgz; rm -rf $(notdir $(var)); fi; popd;)
	@$(foreach var, $(TESTS), echo "\n*******************$(var)********************\n"; pushd .; cd $(var); if [ -e graph.pb ]; then mkdir temp; cd temp; python3 $(KAENA_PATH)/compiler/scripts/tffe --tfpb ../graph.pb --image linspace1; cp -RH sg00/* ../; cd ../; rm -rf temp; fi; popd;)

all:
	@$(foreach var, $(TESTS), $(MAKE) $(notdir $(var));)

sanity:
	@$(foreach var,$(SANITY_TESTS), $(MAKE) $(var);)

inkling:
	@$(foreach var,$(SANITY_TESTS), $(MAKE) $(var); pushd .; cd $(var); $(KAENA_PATH)/compiler/util/run_be_inkling; popd;)

check:
	sanity

%: 
	@echo "******************* $* (tests/$*/log-me.txt) ********************" && cd tests/$* && if [ -e compiler.json ]; then python3 $(KAENA_PATH)/compiler/me/me_main.py --debug 4 --kgraph compiler.json --wavegraph wavegraph.json > log-me.txt && grep -e ERROR -e PASS -e FAIL log-me.txt; fi

unittest:
	#python3 -m unittest layeropt_utils.py 
	python3 -m unittest layeropt_utils.TestShapeDims
	python3 -m unittest layeropt_utils.TestFileParams
	python3 -m unittest layeropt_utils.TestFileMapper

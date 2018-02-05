# Layer-Optimizer tests

help:
	@echo "\nList of tests:"
	@ls layeropt_tests
	@echo ""

layeropt_sanity:
	cd layeropt_tests/0-2conv3_relu/ && python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json

%: 
	cd layeropt_tests/$* && if [ -e out_compiler.json ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph out_compiler.json --wavegraph out_wavegraph.json --dot out_wavegraph.svg; fi
	cd layeropt_tests/$* && if [ -e trivnet_compiler.json ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json --wavegraph trivnet_wavegraph.json --dot trivnet_wavegraph.svg; fi

# Layer-Optimizer tests

help:
	@echo "\nList of tests:"
	@ls layeropt_tests
	@echo ""

layeropt_sanity:
	cd layeropt_tests/0-2conv3_relu/ && python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json

resnet50_conv1:
	cd layeropt_tests/resnet50_conv1 && python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph out_compiler.json

%: 
	cd layeropt_tests/$* && if [ -e out_compiler.json ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph out_compiler.json; fi
	cd layeropt_tests/$* && if [ -e trivnet_compiler.json ]; then python3 $(KAENA_PATH)/compiler/util/layeropt.py --kgraph trivnet_compiler.json; fi

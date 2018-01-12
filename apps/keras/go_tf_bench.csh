#!/bin/csh
foreach i ($KAENA_PATH/../Kaena-external-opensource/images/dog.jpg)
	echo "-------------------- $i ------------------"
	#python3 test_resnet50_keras_inference.py --image=$i --num_loops=200 |& grep -e Keras
	python3 test_resnet50_tf_inference.py --input_layer=input_1 --output_layer=fc1000/Softmax --graph=$KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet50/resnet50_fp32_keras_opt.pb --num_loops=200 --image=$i  |& grep -e TF
	#python3 test_resnet50_keras_inference.py --fp16 --image=$i --num_loops=200 |& grep -e Keras
	#python3 test_resnet50_tf_inference.py --input_layer=input_1 --output_layer=fc1000/Softmax --graph=$KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet50/resnet50_fp16_keras_opt.pb --num_loops=200 --fp16 --image=$i  |& grep -e TF
end

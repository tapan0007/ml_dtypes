#!/bin/csh
foreach i ($KAENA_EXT_PATH/images/*.{jpg,jpeg})
        echo "-------------------- "`basename $i`"------------------"
        #python3 test_resnet50_keras_inference.py --image=$i |& grep -e Keras
        #python3 test_resnet50_tf_inference.py --input_layer=input_1 --output_layer=fc1000/Softmax --graph=$KAENA_EXT_PATH/apps/tf/resnet50_keras/resnet50_fp32_keras_opt.pb --image=$i  |& grep -e TF
        #python3 test_resnet50_keras_inference.py --fp16 --image=$i |& grep -e Keras
        #python3 test_resnet50_tf_inference.py --input_layer=input_1 --output_layer=fc1000/Softmax --graph=$KAENA_EXT_PATH/apps/tf/resnet50_keras/resnet50_fp16_keras_opt.pb --fp16 --image=$i  |& grep -e TF
        python3 test_resnet50_tf_inference.py --input_layer=input_1 --output_layer=fc1000/Softmax --graph=$KAENA_EXT_PATH/apps/tf/resnet50_keras/resnet50_mixed_keras_opt.pb --image=$i  |& grep -e TF
end

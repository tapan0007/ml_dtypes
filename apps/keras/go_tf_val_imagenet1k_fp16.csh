#!/bin/csh
python3 val_resnet50_tf_inference.py --fp16 --input_layer=input_1 --output_layer=fc1000/Softmax --graph=$KAENA_PATH/../Kaena-external-opensource/apps/tf/resnet50_keras/resnet50_fp16_keras_opt.pb | tee $0.log

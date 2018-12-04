import os
import argparse
import time
import tensorflow as tf
import numpy as np

from keras.applications.inception_v3 import InceptionV3
from keras.preprocessing import image
from keras.applications.inception_v3 import preprocess_input, decode_predictions
from keras import backend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    parser.add_argument("--num_loops", type=int, default=1, help="number of inferences to run")
    args = parser.parse_args()

    # set Keras global configurations
    backend.set_learning_phase(0)
    if (args.fp16):
        float_type = 'float16'
    else:
        float_type = 'float32'
    backend.set_floatx(float_type)

    # load pre-trained model using Keras
    model_name = 'inception_v3_%s_keras'%float_type
    model = InceptionV3(weights='imagenet')

    # load image using Keras
    img = image.load_img(args.image, target_size=(args.input_width, args.input_height))
    x = image.img_to_array(img)
    print(x.shape)
    x = np.expand_dims(x, axis=0)
    print(x.shape)
    x = preprocess_input(x)

    # run first prediction using Keras (warmup)
    preds = model.predict(x)

    # run many predictions using Keras (benchmarking)
    start = time.time()
    for i in range(args.num_loops):
        preds = model.predict(x)
    end = time.time()

    # decode the results into a list of tuples (class, description, probability)
    print("\n---------- %s %s ----------"%(model_name, os.path.basename(args.image)))
    print('Keras %s prediction: '%float_type, decode_predictions(preds, top=3)[0])
    if (args.num_loops > 1):
        print("Keras %s msec/inference: %f\n"%(float_type,(end-start)*1000/args.num_loops))

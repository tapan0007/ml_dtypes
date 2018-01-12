import time
import argparse
import sys
import os
import numpy as np
import tensorflow as tf

from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend

def load_graph(model_file):
    graph = tf.Graph()
    graph_def = tf.GraphDef()

    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)

    return graph

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--image", help="image to be processed")
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--num_loops", type=int, default=1, help="number of inferences to run")
    args = parser.parse_args()

    # load graph from file using TF
    graph = load_graph(args.graph)

    # set Keras global configurations
    backend.set_learning_phase(0)
    if (args.fp16):
        float_type = 'float16'
    else:
        float_type = 'float32'
    backend.set_floatx(float_type)

    # load image and preprocess using Keras
    img = image.load_img(args.image, target_size=(args.input_width, args.input_height))
    t = image.img_to_array(img)
    t = np.expand_dims(t, axis=0)
    t = preprocess_input(t)

    # get input and output operation (TF)
    input_name = "import/" + args.input_layer
    output_name = "import/" + args.output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    # warm up using TF
    with tf.Session(graph=graph) as sess:
        results = sess.run(output_operation.outputs[0],
                              {input_operation.outputs[0]: t})

    # main loop using TF, with timer
    start = time.time()
    with tf.Session(graph=graph) as sess:
        for i in range(args.num_loops):
            results = sess.run(output_operation.outputs[0], {input_operation.outputs[0]: t})

    end = time.time()

    # decode the results into a list of tuples (class, description, probability) using Keras
    print("\n---------- %s %s ----------"%(os.path.basename(args.graph), os.path.basename(args.image)))
    print('TF    %s prediction: '%float_type, decode_predictions(results, top=3)[0])
    if (args.num_loops > 1):
        print("TF    %s msec/inference: %f\n"%(float_type,(end-start)*1000/args.num_loops))

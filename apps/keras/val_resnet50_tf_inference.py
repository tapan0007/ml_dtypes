import os
import glob
import tarfile
import time
import timeit
import argparse
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
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
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

    # download 1k validation set
    path = os.environ['HOME']+'/.keras/datasets/val_1000.tar'
    if (not os.path.exists(path)):
        get_file('val_1000.tar', origin='http://data.mxnet.io/mxnet/data/val_1000.tar')
        tar = tarfile.open(path)
        tar.extractall()
        tar.close()

    # load image using Keras
    top5 = 0
    top1 = 0
    cnt = 0
    time = 0
    aggtime = 0
    labels = np.loadtxt("val_1000/label", dtype=int, usecols=0)
    for file in glob.glob("val_1000/*.jpg"):
        # get the file name without extension
        img_id_txt = os.path.splitext(os.path.basename(file))[0]
        if (img_id_txt.isdigit()):
            img_id = int(img_id_txt)
            label = labels[img_id]
            cnt += 1

            # load image and preprocess using Keras
            img = image.load_img(file, target_size=(args.input_width, args.input_height))
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
                start = timeit.default_timer()
                results = sess.run(output_operation.outputs[0],
                                      {input_operation.outputs[0]: t})
                time = timeit.default_timer()-start
                aggtime += time

            # take the top 5
            top_indices = results[0].argsort()[-5:][::-1]
            if (len([i for i in top_indices if i==label])>0):
                top5 += 1
            if (label == top_indices[0]):
                top1 += 1

            # decode the results into a list of tuples (class, description, probability) using Keras
            print("\n---------- %s %s ----------"%(os.path.basename(args.graph), os.path.basename(file)))
            print('TF    %s prediction: '%float_type, decode_predictions(results, top=3)[0])
            print('cnt %d top1 %d top5 %d top1 %f top5 %f time %f aggtime %f avgtime %f'%(cnt, top1, top5, top1/cnt, top5/cnt, time, aggtime, aggtime/cnt))

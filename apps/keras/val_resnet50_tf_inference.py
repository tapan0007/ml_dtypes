import os
import glob
import timeit
import argparse
import numpy as np
import tensorflow as tf
import mxnet as mx
import boto3
import botocore

from urllib.parse import urlparse
from keras.preprocessing import image
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend
from val_resnet50_keras_inference import download_set

def load_graph(model_file, s3_profile='kaena'):
    graph = tf.Graph()
    graph_def = tf.GraphDef()
    if model_file.startswith('s3://'):
        parse = urlparse(model_file)
        bucket = parse.netloc
        model_name = parse.path[1:]
        model_path = os.path.abspath(model_name)
        if not os.path.exists(model_path):
            print("INFO: downloading {} to {}..."
                .format(model_file, os.path.abspath('.')))
            boto3_sess = boto3.session.Session(profile_name=s3_profile)
            s3 = boto3_sess.resource('s3')
            try:
                s3.Bucket(bucket).download_file(model_name, model_path)
            except botocore.exceptions.ClientError as e:
                if e.response['Error']['Code'] == "404":
                    print("ERROR: The object does not exist.")
                else:
                    raise
        model_file = model_path
    with open(model_file, "rb") as f:
        graph_def.ParseFromString(f.read())
    with graph.as_default():
        tf.import_graph_def(graph_def)
    return graph, model_file

def create_data_iter(dataset, batch_size, img_target_size):
    # download validation set
    path = download_set(dataset)

    # create mxnet data iterator
    if (dataset == "1k"):
        labels = np.loadtxt(path + "/label", dtype=int, usecols=0)
        img_dict = {}
        for filename in glob.glob(path + "/*.jpg"):
            # get the file name without extension
            img_id_txt = os.path.splitext(os.path.basename(filename))[0]
            if img_id_txt.isdigit():
                img_id = int(img_id_txt)
                label = labels[img_id]
                img = image.load_img(filename, target_size=img_target_size)
                t = image.img_to_array(img)
                t = np.transpose(t, (2, 0, 1)) # HWC --> CHW
                img_dict[img_id] = t
        images = np.asarray([img_dict[i] for i in sorted(img_dict.keys())])
        data_iter = mx.io.NDArrayIter(
                data=images, label=labels, batch_size=batch_size)
    else:
        data_iter = mx.io.ImageRecordIter(
                path_imgrec=path,
                data_shape=(3, 224, 224),
                batch_size=batch_size)
    return data_iter

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph", help="graph/model to be executed; "
        "supports s3://kaena-nn-models/resnet50_fp32_keras_opt3.pb like syntax")
    parser.add_argument("--s3_profile", default='kaena', help="aws s3 profile name")
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--dataset", default="1k", help="Dataset: one of 1k, 5k, or 50k")
    parser.add_argument("--stop_at_idx", type=int, default=-1, help="stop processing at image index; -1 to do the entire set of images")
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    parser.add_argument("--input_layer", help="name of input layer")
    parser.add_argument("--output_layer", help="name of output layer")
    parser.add_argument("--batch_size", type=int, default=32, help="batch size for inference")
    args = parser.parse_args()

    # load graph from file using TF
    graph, _ = load_graph(args.graph, args.s3_profile)

    # get input and output operation (TF)
    input_name = "import/" + args.input_layer
    output_name = "import/" + args.output_layer
    input_operation = graph.get_operation_by_name(input_name);
    output_operation = graph.get_operation_by_name(output_name);

    # set Keras global configurations
    backend.set_learning_phase(0)
    if (args.fp16):
        float_type = 'float16'
    else:
        float_type = 'float32'
    backend.set_floatx(float_type)

    # create mxnet data iter
    data_iter = create_data_iter(args.dataset, args.batch_size,
        img_target_size=(args.input_width, args.input_height))

    # compute classification accuracy
    top5 = 0
    top1 = 0
    cnt = 0
    time = 0
    aggtime = 0
    with tf.Session(graph=graph) as sess:
        for batch in data_iter:
            t = batch.data[0].asnumpy()
            t = np.transpose(t, (0, 2, 3, 1)) # NCHW -> NHWC
            t = preprocess_input(t)
            start = timeit.default_timer()
            results = sess.run(output_operation.outputs[0],
                                  {input_operation.outputs[0]: t})
            time = timeit.default_timer() - start
            aggtime += time

            results = results[:len(results)-batch.pad]
            # take the top 5
            for result, label in zip(results, batch.label[0].asnumpy()):
                cnt += 1
                top_indices = result.argsort()[-5:][::-1]
                if (len([i for i in top_indices if i==label])>0):
                    top5 += 1
                if (label == top_indices[0]):
                    top1 += 1

                # decode the results into a list of tuples (class, description, probability) using Keras
                print("\n---------- %s image %d label %d ----------"%(os.path.basename(args.graph), cnt, label))
                print('TF    %s prediction: '%float_type, decode_predictions(result[np.newaxis, ...], top=3)[0])
                print('cnt %d top1 %d top5 %d top1 %f top5 %f time %f aggtime %f avgtime %f'%(cnt, top1, top5, top1/cnt, top5/cnt, time, aggtime, aggtime/cnt))
                if cnt == args.stop_at_idx:
                    exit(0)

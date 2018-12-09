import os
import re
import glob
import tarfile
import urllib
import time
import timeit
import argparse
import numpy as np
import tensorflow as tf
import mxnet as mx

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend

def download_set(dataset):
    if (dataset == "1k"):
        recfile = 'val_1000.tar'
    elif (dataset == "5k"):
        recfile = 'val-5k-256.rec'
    elif (dataset == "50k"):
        recfile = 'val_256_q90.rec'
    if (not os.path.exists(recfile)):
        urllib.request.urlretrieve('http://data.mxnet.io/mxnet/data/' + recfile, recfile)
        if re.search(".tar", recfile):
            tar = tarfile.open(recfile)
            tar.extractall()
            tar.close()
    recname = re.sub("\.tar", "", recfile)
    return recname

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action='store_true')
    parser.add_argument("--dataset", default="1k", help="Dataset: one of 1k, 5k, or 50k")
    parser.add_argument("--stop_at_idx", type=int, default=-1, help="stop processing at image index; -1 to do the entire set of images")
    parser.add_argument("--input_height", type=int, default=224, help="input height")
    parser.add_argument("--input_width", type=int, default=224, help="input width")
    args = parser.parse_args()

    # set Keras global configurations
    backend.set_learning_phase(0)
    if (args.fp16):
        float_type = 'float16'
    else:
        float_type = 'float32'
    backend.set_floatx(float_type)

    # load pre-trained model using Keras
    model_name = 'resnet50_%s_keras'%float_type
    model = ResNet50(weights='imagenet')

    # load image using Keras
    top5 = 0
    top1 = 0
    cnt = 0
    time = 0
    aggregate_time = 0

    # download validation set
    path = download_set(args.dataset)

    if (args.dataset == "1k"):
        labels = np.loadtxt(path + "/label", dtype=int, usecols=0)
        for file in glob.glob(path + "/*.jpg"):
            # get the file name without extension
            img_id_txt = os.path.splitext(os.path.basename(file))[0]
            if (img_id_txt.isdigit()):
                img_id = int(img_id_txt)
                label = labels[img_id]
                cnt += 1

                img = image.load_img(file, target_size=(args.input_width, args.input_height))
                x = image.img_to_array(img)
                x = np.expand_dims(x, axis=0)
                x = preprocess_input(x)

                # run first prediction using Keras (warmup)
                start = timeit.default_timer()
                preds = model.predict(x)
                time = timeit.default_timer()-start
                aggregate_time += time

                # take the top 5
                top_indices = preds[0].argsort()[-5:][::-1]
                if (len([i for i in top_indices if i==label])>0):
                    top5 += 1
                if (label == top_indices[0]):
                    top1 += 1

                # decode the results into a list of tuples (class, description, probability)
                print("\n---------- %s %s ----------"%(model_name, os.path.basename(file)))
                print('Keras %s prediction: '%float_type, decode_predictions(preds, top=3)[0])
                print('count %d top1 %d top5 %d top1 %f top5 %f time %f aggregate_time %f average_time %f'%(cnt, top1, top5, top1/cnt, top5/cnt, time, aggregate_time, aggregate_time/cnt))
                if (cnt == args.stop_at_idx):
                    break
    else:        
        data_iter = mx.io.ImageRecordIter (
                path_imgrec=path,
                data_shape=(3, 224, 224),
                batch_size=1
                )

        #for file in glob.glob("val_1000/*.jpg"):
        for batch in data_iter:
            # get the file name without extension
            label = batch.label[0].asnumpy()
            cnt += 1

            x = batch.data[0].asnumpy()
            x = np.transpose(x, (0,2,3,1)) # NCHW -> NHWC
            x = preprocess_input(x)

            # run first prediction using Keras (warmup)
            start = timeit.default_timer()
            preds = model.predict(x)
            time = timeit.default_timer()-start
            aggregate_time += time

            # take the top 5
            top_indices = preds[0].argsort()[-5:][::-1]
            if (len([i for i in top_indices if i==label])>0):
                top5 += 1
            if (label == top_indices[0]):
                top1 += 1

            # decode the results into a list of tuples (class, description, probability)
            print("\n---------- %s image %d label %d ----------"%(model_name, cnt, label))
            print('Keras %s prediction: '%float_type, decode_predictions(preds, top=3)[0])
            print('count %d top1 %d top5 %d top1 %f top5 %f time %f aggregate_time %f average_time %f'%(cnt, top1, top5, top1/cnt, top5/cnt, time, aggregate_time, aggregate_time/cnt))
            if (cnt == args.stop_at_idx):
                break

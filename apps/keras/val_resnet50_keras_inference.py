import os
import glob
import tarfile
import argparse
import time
import timeit
import tensorflow as tf
import numpy as np

from keras.applications.resnet50 import ResNet50
from keras.preprocessing import image
from keras.utils.data_utils import get_file
from keras.applications.resnet50 import preprocess_input, decode_predictions
from keras import backend

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp16", action='store_true')
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

            img = image.load_img(file, target_size=(args.input_width, args.input_height))
            x = image.img_to_array(img)
            x = np.expand_dims(x, axis=0)
            x = preprocess_input(x)

            # run first prediction using Keras (warmup)
            start = timeit.default_timer()
            preds = model.predict(x)
            time = timeit.default_timer()-start
            aggtime += time

            # take the top 5
            top_indices = preds[0].argsort()[-5:][::-1]
            if (len([i for i in top_indices if i==label])>0):
                top5 += 1
            if (label == top_indices[0]):
                top1 += 1

            # decode the results into a list of tuples (class, description, probability)
            print("\n---------- %s %s ----------"%(model_name, os.path.basename(file)))
            print('Keras %s prediction: '%float_type, decode_predictions(preds, top=3)[0])
            print('cnt %d top1 %d top5 %d top1 %f top5 %f time %f aggtime %f avgtime %f'%(cnt, top1, top5, top1/cnt, top5/cnt, time, aggtime, aggtime/cnt))

#!/usr/bin/env python3

import sys
import argparse
import numpy as np
from PIL import Image
from keras.applications.resnet50 import preprocess_input


def _img_to_numpy_array(imgFile, size, dtype):
    """ Coverts given image to numpy array.

    :param imgFile: Image file path.
    :return: numpy.array.
    """
    print('Loading: ' + imgFile)
    img = Image.open(imgFile)
    return np.array(img.resize(size), dtype=dtype)


def preprocess(imgFiles, ifmapFile, dtype, height, width):
    """ Pre-processes given image files using keras.resnet50

    :param imgFiles: Image files to be processed.
    :param ifmapFile: Output file location.
    :return:
    """

#    npArrList = [np.array(_img_to_numpy_array(img, (224, 224), dtype)) for img in imgFiles]
    print ("height = %s width = %s" %(height, width))
    npArrList = [np.array(_img_to_numpy_array(img, (int(height), int(width)), dtype)) for img in imgFiles]
    npArr = np.array(npArrList)
#    fmap = npArr.reshape(len(imgFiles), 224, 224, 3)
    fmap = npArr.reshape(len(imgFiles), int(height), int(width), 3)
    fmap = preprocess_input(fmap)
    np.save(ifmapFile, fmap)
    print('Wrote: ' + ifmapFile)


def main():
    print("\nINFO: started as  ", " ".join(sys.argv), flush=True)

    parser = argparse.ArgumentParser()
    parser.add_argument('--inputs', help='input image files',
                        dest="inputs", default=[], nargs='+', required=True)
    parser.add_argument('--output', help='Output fmap npy file',
                        dest="output", required=True)
    parser.add_argument('--data-type', help='datatype', choices=['fp16', 'fp32'],
                        required=True)
    parser.add_argument('--input_height', help='input height', default=299)
    parser.add_argument('--input_width', help='input width', default=299)

    args = parser.parse_args()
    if args.data_type == 'fp16':
        datatype = np.float16
    else:
        datatype = np.float32
    preprocess(args.inputs, args.output, datatype, args.input_height, args.input_width)


if __name__ == "__main__":
    main()

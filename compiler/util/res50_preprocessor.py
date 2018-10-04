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


def preprocess(imgFiles, ifmapFile, dtype, size):
    """ Pre-processes given image files using keras.resnet50

    :param imgFiles: Image files to be processed.
    :param ifmapFile: Output file location.
    :return:
    """

    npArrList = [np.array(_img_to_numpy_array(img, (size, size), dtype)) for img in imgFiles]
    npArr = np.array(npArrList)
    fmap = npArr.reshape(len(imgFiles), size, size, 3)
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
    parser.add_argument('--size', help='Width/Height of the square output image',
                        default=224, type=int)

    args = parser.parse_args()
    if args.data_type == 'fp16':
        datatype = np.float16
    else:
        datatype = np.float32
    preprocess(args.inputs, args.output, datatype, args.size)


if __name__ == "__main__":
    main()

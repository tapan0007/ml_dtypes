#!/usr/bin/env python3

import os
import sys
import argparse
sys.path.insert(0, os.environ["KAENA_PATH"] + "/compiler/me")
from me_utils import pad_and_split_file

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file", required=True, help="Input numpy file to process. Currently, only NCHW format supported.")
    parser.add_argument("--format", default="NCHW", help="Format of input file. Currently, only NCHW format supported.")
    parser.add_argument("--stride", default=2, help="Stride value of the convolution requiring replication.")
    parser.add_argument("--padding", default="[ [ 0, 0 ], [ 0, 0 ], [ 2, 3 ], [ 2, 3 ] ]", help="Padding info from compiler.json, expressed as a string; i.e. \"[ [ 0, 0 ], [ 0, 0 ], [ 2, 3 ], [ 2, 3 ] ]\"")
    args = parser.parse_args()

    if args.format != "NCHW":
        raise RuntimeError("Please convert numpy input file to NCHW format first.")

    if not os.path.isfile(args.input_file):
        raise RuntimeError("%s doesn't exist"%args.input_file)
    else:
        print("Processing %s with format %s, stride %d, and padding %s"%(args.input_file, args.format, args.stride, args.padding))

    try:
        padding = eval(args.padding)
    except Exception as e:
        print("ERROR while trying to parse padding string \"%s\":"%args.padding)
        print(e)
        exit(1)

    (padN, padS) = padding[2]
    (padW, padE) = padding[3]

    (file_name, new_shape) = pad_and_split_file(
                                args.input_file,
                                args.format,
                                args.stride,
                                padW, padE,
                                padN, padS)

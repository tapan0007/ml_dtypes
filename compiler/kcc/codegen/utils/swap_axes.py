#!/usr/bin/python

import argparse
import random
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('-i',type=str, default="stdout", help='array file input')
parser.add_argument('-o',type=str, default="stdout", help='array file output')
parser.add_argument('-s',type=int, nargs=2, required=True, help='axes to swap (space seperated list)')
args = parser.parse_args()

i = np.load(args.i)
o = i.swapaxes(*args.s).copy(order='C')

if args.o == "stdout":
    print o
else:
    np.save(args.o, o)


#!/usr/bin/python

import argparse
import random
import os
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument('ifiles', type=str,  nargs='+', help='input numpy array pyc files')
args = parser.parse_args()


np.set_printoptions(threshold=np.nan)

for ifile in args.ifiles:
    nmap = np.load(ifile)
    print "{} is of type {} and shape {}".format(ifile,  nmap.dtype, nmap.shape)
    print nmap.astype(str)


#!/usr/bin/python

import argparse
import random
import os
import numpy as np

def diff(_n0, _n1):
    np.set_printoptions(threshold=np.nan)
    n0 = np.load(_n0)
    n1 = np.load(_n1)
    if n0.dtype == np.dtype(np.float32):
        np.testing.assert_allclose(n0, n1,  err_msg=err_msg)
    else:
        np.testing.assert_array_equal(n0, n1)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('ifiles', type=str,  nargs=2, help='input numpy array pyc files')
    args = parser.parse_args()
    diff(args.ifiles[0], args.ifiles[1])





#!/usr/bin/python
import os
import sys


base_dir = '../../../'
paths = [base_dir + 'golden_model', base_dir]
for path in paths:
    apath = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if not apath in sys.path:
        sys.path.insert(1, apath)

TEST_CONVOLVE='../../tcc/test/convolve/test_convolve'
TEST_POOL='../../tcc/test/pool/test_pool'
SIM = '../../sim/sim'

import numpy as np
import uuid
from utils import make_array

def rand_fname(preq, dtype, dims):
    fname = "{}_{}_{}_{}.npy".format(preq, dtype, "x".join([str(d) for d in dims]), str(uuid.uuid4())[:8])
    mypath = "{}/{}".format('input/random/', fname)
    assert(not os.path.exists(mypath) and "path exists, clean up input dir?")
    return mypath

def randf(preq, dtype, dims):
    mypath = rand_fname(preq, dtype, dims)
    A = make_array.create(dtype, None, dims)
    np.save(mypath, A)
    return mypath


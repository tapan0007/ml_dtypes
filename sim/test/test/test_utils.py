#!/usr/bin/python
import os
import sys


base_dir = '../../../'
paths = [base_dir + 'golden_model', base_dir]
for path in paths:
    apath = os.path.abspath(os.path.join(os.path.dirname(__file__), path))
    if not apath in sys.path:
        sys.path.insert(1, apath)

TEST_CONVOLVE='../../tcc/test/convolve/test_convolve.bin'
TEST_POOL='../../tcc/test/pool/test_pool.bin'
TEST_ACTIVATION='../../tcc/test/activation/test_activation.bin'
SIM = '../../sim/sim'

import numpy as np
import uuid
from utils import make_array

def rand_fname(preq, dtype, dims):
    fname = "{}_{}_{}_{}.npy".format(preq, dtype, "x".join([str(d) for d in dims]), str(uuid.uuid4())[:8])
    mypath = "{}/{}".format('input/random/', fname)
    assert(not os.path.exists(mypath) and "path exists, clean up input dir?")
    return mypath

def randf(preq, dtype, dims, vmin = None, vmax = None):
    if vmin is None:
        if 'int' in dtype:
            vmin = np.iinfo(dtype).min/10
        else:
            vmin = np.finfo(dtype).min/1000
    if vmax is None:
        if 'int' in dtype:
            vmax = np.iinfo(dtype).max/10
        else:
            vmax = np.finfo(dtype).max/1000
    mypath = rand_fname(preq, dtype, dims)
    A = make_array.create(dtype, None, dims, vmin, vmax)
    np.save(mypath, A)
    return mypath


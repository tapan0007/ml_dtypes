#!/usr/bin/python 
"""

"""
import os
import sys
import random
import numpy as np
import uuid
sys.path.append(os.path.realpath('../utils'))
import make_array

input_dir = "test/input/random"

def randf(preq, dtype, dims):
    fname = "{}_{}_{}_{}.npy".format(preq, dtype, "x".join([str(d) for d in dims]), str(uuid.uuid4())[:8])
    mypath = "{}/{}".format(input_dir, fname)
    assert(not os.path.exists(mypath) and "path exists, clean up input dir?")
    if 'int' in dtype:
        vmin =   0 # must be positive because tf goes uint80>int32
        vmax =  10
    else:
        vmin = 0.0
        vmax = 1.0
    A = make_array.create(dtype, None, dims, vmin, vmax)
    np.save(mypath, A)
    return mypath 


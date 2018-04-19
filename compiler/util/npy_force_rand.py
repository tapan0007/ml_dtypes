#!/usr/bin/env python3

import os
import numpy as np
import sys

x=np.load(sys.argv[1])
y=np.random.rand(x.size)
z=y.astype(x.dtype)
np.save(sys.argv[1],z.reshape(x.shape))


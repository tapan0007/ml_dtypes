#!/usr/bin/env python3

import os
import numpy as np
import sys

x=np.load(sys.argv[1])
y=np.ones(x.shape, x.dtype)
np.save(sys.argv[1],y)


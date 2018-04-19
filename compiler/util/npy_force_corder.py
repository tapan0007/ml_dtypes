#!/usr/bin/env python3

import os
import numpy as np
import sys

x=np.load(sys.argv[1])
y=np.copy(x, order='C')
np.save(sys.argv[1],y)


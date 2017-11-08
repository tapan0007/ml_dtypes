# Numpy utilities

import numpy as np

class NpUtils:
  @staticmethod
  def showNpyFile(prefixText, fileName, viewDtype=None):
    arr = np.load(fileName)
    if viewDtype == None:
      viewDtype = arr.dtype
    print(prefixText, "  ", fileName, "\n", arr.view(dtype=viewDtype))


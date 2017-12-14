# Numpy utilities

import numpy as np

class NpUtils:
  @staticmethod
  def showNpyFile(prefixText, fileName, viewDtype=None):
    arr = np.load(fileName)
    if viewDtype == None:
      viewDtype = arr.dtype
    print(prefixText, "  ", fileName, "\n", arr.view(dtype=viewDtype))

  @staticmethod
  def dtypeToStr(dType):
    if dType == np.float32:
      dTypeStr = "float32"
    if dType == np.float16:
      dTypeStr = "float16"
    else:
      raise NameError("Unsupported input node data type %s" % str(dType))
    return dTypeStr
    

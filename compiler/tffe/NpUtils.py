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

  # Custom comparison utility with lower sensitivity by considering overall value range
  #   Spec https://sim.amazon.com/issues/kaena-181
  # Modes:
  #   py - like python all close but still has the verbose option
  #   max - use max B for relative difference -> very loose match criteria
  @staticmethod
  def allclose(a, b, rtol=1e-05, atol=1e-08, equal_nan_inf=False, verbose=False, mode='max'):
    # b is gold/refernce, a is new
    if not a.shape == b.shape:
      return False

    # Range criteria selection
    # Several large convolutions did not pass using range or average
    #   aver: 1-1conv7_64  1-1conv9_64  3-rn50_pool2  range : 0-1ap7x7
    #bRange = abs(np.amax(b) - np.amin(b))
    #bRange = abs(np.average(b))
    bRange = None
    if mode == 'max':
      bRange = abs(np.amax(b))
    for index, bval in np.ndenumerate(b):
      aval = a[index]
      bval = b[index]
      if bval == np.nan or bval == np.inf:
        if not (aval == bval and equal_nan_inf):
          return False
      else:
        if mode == 'py':
          bRange = abs(bval)
        aval = float(aval)
        bval = float(bval)
        if not (abs(aval - bval) <= (atol + rtol * bRange)):
          if verbose:
            print("INFO: NpUtils::allclose comparison failed  abs(%g - %g) <= (%g + %g * %g)" % (aval, bval, atol, rtol, bRange))
          return False
    return True



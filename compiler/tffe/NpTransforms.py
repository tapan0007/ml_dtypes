# Numpy file transformation definitions and services

import numpy as np

class NpTrans:
  # See spec for method  genCompilerPy
  TF = "TF"
  SIM = "SIM"
  TF2SIM = "TF2SIM"
  Fmaps   = "Fmaps"
  Weights = "Weights"
  NHWC = "NHWC"
  NCHW = "NCHW"
  NCHW = "NCHW"
  MCRS = "MCRS"
  Formats = {
    TF : {
      Fmaps   : NHWC,
      Weights : NCHW
      },
    SIM : {
      Fmaps   : NCHW,
      Weights : MCRS
      }
    }
  Transforms = {
    TF2SIM : {
      Fmaps   : [ [1,2], [1,3] ],
      Weights : [ [0,3], [1,2], [2,3] ]
      }
    }
  # Ulility function to convert npy files, can be moved out of the graph later
  @staticmethod
  def copyNpyFileAs(npFile, destFormat, transformList):
    arr = np.load(npFile)
    for transform in transformList:
      arr = np.swapaxes(arr, *transform)
    npFileDest = npFile.replace(".npy", "_" + destFormat + ".npy")
    np.save(npFileDest, arr)
    return(npFileDest)


# Numpy file transformation definitions and services

import numpy as np

def calcTransform(sf, st):
  assert(len(sf) == len(st))
  transform = [sf.find(c) for c in list(st)]
  #print("DEBUG: transform=", transform)
  assert(transform[0] >= 0)
  return(transform)

class NpTrans:
  # See spec for method  genCompilerPy
  TF = "TF"
  SIM = "SIM"
  TF2SIM = "TF2SIM"
  Fmaps   = "Fmaps"
  Weights = "Weights"
  NHWC = "NHWC"
  NCHW = "NCHW"
  RSCM = "RSCM"
  MCRS = "MCRS"
  Formats = {
    TF : {
      Fmaps   : NHWC,
      Weights : RSCM
      },
    SIM : {
      Fmaps   : NCHW,
      Weights : MCRS
      }
    }
  Transforms = {
    TF2SIM : {
      Fmaps   : calcTransform(Formats[TF][Fmaps],   Formats[SIM][Fmaps]),
      Weights : calcTransform(Formats[TF][Weights], Formats[SIM][Weights])
      }
    }
  # Ulility function to convert npy files, can be moved out of the graph later
  @staticmethod
  def copyNpyFileAs(npFile, destFormat, transform):
    arr = np.load(npFile)
    arr = np.transpose(arr, transform)
    npFileDest = npFile.replace(".npy", "_" + destFormat + ".npy")
    np.save(npFileDest, arr)
    return(npFileDest)


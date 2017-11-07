# Numpy file transformation definitions and services
#
# It allows to specify framework (e.g. TF) formats for weights
# and fmaps (e.g. NHWC) and destination platform (e.g. SIM - Inlikng
# simulator)
# It computes the transformations upon import, and at runtime performs
# the data format conversions.

import numpy as np

def calcTransform(sf, st):
  assert(len(sf) == len(st))
  transform = [sf.find(c) for c in list(st)]
  #print("DEBUG: transform=", transform)
  assert(all(i >=0 for i in transform))
  return(transform)

class NpTrans:
  # See spec for method  genCompilerPy
  for c in ["TF", "SIM", "Fmaps", "Weights", "NHWC", "NCHW", "RSCM", "MCRS"]:
    exec("%s = '%s'" %(c, c))
  
  # Define tensorFlow (TF) to Inkling simulator (SIM) translation
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
  Transforms = {}
  for src in [TF, SIM]:
    Transforms[src] = {}
    for dst in [TF, SIM]:
      Transforms[src][dst] = {}
      for d in [Fmaps, Weights]:
        Transforms[src][dst][d] = calcTransform(Formats[src][d], Formats[dst][d])
  # Ulility function to convert npy files, returns new file name and the destination format
  @staticmethod
  def copyNpyFileAs(npFile, srcPlat, dstPlat, dataFlavor):
    dstFormat = NpTrans.Formats[dstPlat][dataFlavor]
    transform = NpTrans.Transforms[srcPlat][dstPlat][dataFlavor]
    arr = np.load(npFile)
    arr = np.transpose(arr, transform)
    npFileDest = npFile.replace(".npy", "_" + dstFormat + ".npy")
    np.save(npFileDest, arr)
    return(npFileDest, dstFormat)


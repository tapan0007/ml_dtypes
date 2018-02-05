# Copyright (C) 2017, Amazon.com. All Rights Reserved
#
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
  for c in ["TF", "SIM", "Fmaps", "Weights", "NHWC", "NCHW", "RSCM", "MCRS", "CRSM"]:
    exec("%s = '%s'" %(c, c))
  
  # Define tensorFlow (TF) to Inkling simulator (SIM) translation
  # TPB/Inkling(SIM) data order:
  #   Per Dana - The CRSM is the best order for HW performance.  Actually, you can
  #   give it in any order you want, just know that what Inkling does with the
  #   instruction: put the left-most dimension across the partition rows and the
  #   remaining dimensions sequentially into a row.
  Formats = {
    TF : {
      Fmaps   : NHWC,
      Weights : RSCM
      },
    SIM : {
      Fmaps   : NCHW,
      #Weights : MCRS
      Weights : CRSM
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
  def copyNpyFileAs(npFile, srcPlat, dstPlat, dataFlavor, srcShape=None):
    dstFormat = NpTrans.Formats[dstPlat][dataFlavor]
    transform = NpTrans.Transforms[srcPlat][dstPlat][dataFlavor]
    arr = np.load(npFile)
    if srcShape != None:
      arr = arr.reshape(srcShape)
    arr = np.transpose(arr, transform)
    npFileDest = npFile.replace(".npy", "_" + dstFormat + ".npy")
    np.save(npFileDest, arr)
    return(npFileDest, dstFormat)

  @staticmethod
  def reorderShape(shapeArr, srcPlat, dstPlat, dataFlavor):
    dstFormat = NpTrans.Formats[dstPlat][dataFlavor]
    transform = NpTrans.Transforms[srcPlat][dstPlat][dataFlavor]
    reorderedShape = [shapeArr[i] for i in transform]
    #print("DEBUG: shape %s %s -> %s %s " %(srcPlat, shapeArr, dstPlat, reorderedShape))
    return(reorderedShape)
  
  @staticmethod
  def subShape(shapeArr, subShapeFormat, srcPlat, dataFlavor):
    srcFormat = NpTrans.Formats[srcPlat][dataFlavor]
    subShape = []
    for c in subShapeFormat:
      for i in range(len(srcFormat)):
        if srcFormat[i] == c:
          subShape.append(shapeArr[i])
    #print("DEBUG: shape %s %s -> subshape %s %s" %(srcPlat, shapeArr, subShapeFormat, subShape))
    return(subShape)
  
  @staticmethod
  def cShapeToNHWC(shape):
    tmpShape = [1, 1, 1] + shape
    return tmpShape[-4:]
  
  
  

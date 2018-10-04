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
import json

def calcTransform(sf, st):
  assert(len(sf) == len(st))
  transform = [sf.find(c) for c in list(st)]
  #print("DEBUG: transform=", transform)
  assert(all(i >=0 for i in transform))
  return(transform)

class NpTrans:
  # See spec for method  genCompilerPy
  for c in ["TF", "SIM", "Fmaps", "Weights", "WeightsTrans", "NHWC", "NCHW", "RSCM", "MCRS", "CRSM", "MRSC", "C", "NC", "NW", "HNC", "HNWC", "CM", "NWC"]:
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
      Weights : RSCM,
      WeightsTrans : RSCM
      },
    SIM : {
      Fmaps   : NCHW,
      #Weights : MCRS
      Weights : CRSM,
      WeightsTrans : MRSC
      }
    }
  Transforms = {}
  for src in [TF, SIM]:
    Transforms[src] = {}
    for dst in [TF, SIM]:
      Transforms[src][dst] = {}
      for d in [Fmaps, Weights, WeightsTrans]:
        Transforms[src][dst][d] = calcTransform(Formats[src][d], Formats[dst][d])

  # Ulility function to convert npy files given the precise format spec, returns new file name and the destination format
  @staticmethod
  def formatNpyFileAs(npFile, srcFormat, dstFormat, outFile=None, dstShape=None):
    arr = np.load(npFile)
    assert len(srcFormat) == len(arr.shape)
    srcShape = arr.shape
    sf = srcFormat
    if len(srcFormat) > len(dstFormat):
      sf = ''
      for c in reversed(list(srcFormat)):
        if not c in dstFormat:
          axis = srcFormat.index(c)
          assert srcShape[axis] == 1
          arr = np.squeeze(arr, axis)
        else:
          sf = c + sf
    elif len(srcFormat) < len(dstFormat):
      for c in list(dstFormat):
        if not c in srcFormat:
          arr = np.expand_dims(arr, 0)
          sf = c + sf
    assert sorted(list(sf)) == sorted(list(dstFormat))
    transform = calcTransform(sf, dstFormat)
    arr = np.transpose(arr, transform)
    if not dstShape == None:
      assert arr.size == np.empty(dstShape).size
      arr = arr.reshape(dstShape)
    if outFile == None:
      npFileDest = npFile.replace(".npy", "_" + dstFormat + ".npy")
    else:
      npFileDest = outFile
    np.save(npFileDest, np.ascontiguousarray(arr))
    return(npFileDest, arr.shape)

  # Ulility function to convert npy files, returns new file name and the destination format
  @staticmethod
  def copyNpyFileAs(npFile, srcPlat, dstPlat, dataFlavor, srcShape=None, outFile=None, dstShape=None):
    dstFormat = NpTrans.Formats[dstPlat][dataFlavor]
    transform = NpTrans.Transforms[srcPlat][dstPlat][dataFlavor]
    arr = np.load(npFile)
    if srcShape != None:
      arr = arr.reshape(srcShape)
    arr = np.transpose(arr, transform)
    if not dstShape == None:
      assert arr.size == np.empty(dstShape).size
      arr = arr.reshape(dstShape)
    if outFile == None:
      npFileDest = npFile.replace(".npy", "_" + dstFormat + ".npy")
    else:
      npFileDest = outFile
    np.save(npFileDest, np.ascontiguousarray(arr))
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

  @staticmethod
  def nwShapeToNHWC(shape):
    assert len(shape) == 2
    tmpShape = [shape[0], 1, shape[1], 1]
    return tmpShape
  
  @staticmethod
  def ncShapeToNHWC(shape):
    assert len(shape) == 2
    tmpShape = [shape[0], 1, 1, shape[1]]
    return tmpShape
  
  @staticmethod
  def nwcShapeToNHWC(shape):
    assert len(shape) == 3
    tmpShape = [shape[0], 1, shape[1], shape[2]]
    return tmpShape

  @staticmethod
  def hncShapeToHNWC(shape):
    assert len(shape) == 3
    tmpShape = [shape[0], shape[1], 1, shape[2]]
    return tmpShape

  @staticmethod
  def cmShapeToRSCM(shape):
    assert len(shape) == 2
    tmpShape = [1, 1, shape[0], shape[1]]
    return tmpShape

  @staticmethod
  def reshapeFilePerRefFile(npyFile, refShapeFile):
    arr = np.load(npyFile)
    refArr = np.load(refShapeFile)
    assert refArr.size == arr.size
    if not refArr.shape == arr.shape:
      #print("DEBUG: reshapeFilePerRefFile reshaped  %s  %s -> %s  based on shape of  %s"
      #      % (npyFile, arr.shape, refArr.shape, refShapeFile))
      arr = arr.reshape(refArr.shape)
      np.save(npyFile, np.ascontiguousarray(arr))
    #else:
      #print("DEBUG: reshapeFilePerRefFile no reshape was needed for  %s" % npyFile)


# Mapping of tensor, layer name - structure is in json format
class TensorFormat:
  def __init__(self, tensorName, layerName, tfFile, tfFormat, simFile, simFormat, isConst):
    self.tensorName = tensorName
    self.layerName = layerName
    self.tfFile = tfFile
    self.tfFormat = tfFormat
    self.simFile = simFile
    self.simFormat = simFormat
    self.isConst = isConst
  def asJsonDict(self):
    jsonData = {}
    jsonData["tensor_name"] = self.tensorName
    jsonData["layer_name"] = self.layerName
    jsonData["tf_file"] = self.tfFile
    jsonData["tf_format"] = self.tfFormat
    jsonData["sim_file"] = self.simFile
    jsonData["sim_format"] = self.simFormat
    jsonData["is_const"] = self.isConst
    return jsonData

class TensorFormatMap:
  def __init__(self):
    self.tensors = {}
  def add(self, tensorname, tensorFormat):
    self.tensors[tensorname] = tensorFormat
  def writeJson(self, outFile):
    jsonData = {}
    for tensorname, tensorFormat in self.tensors.items():
      jsonData[tensorname] = tensorFormat.asJsonDict()
    with open(outFile, "w") as f:
      s = json.dumps(jsonData, indent=2, sort_keys=True)
      f.write(s)
  def readJson(self, inFile):
    with open(inFile) as fh:
      jsonData = json.load(fh)
    for tensorname, tfj in jsonData.items():
      self.tensors[tensorname] = TensorFormat(tfj["tensor_name"], tfj["layer_name"], tfj["tf_file"], tfj["tf_format"], tfj["sim_file"], tfj["sim_format"], tfj["is_const"])
  def get(self, tensorName):
    return self.tensors.get(tensorName, None)
  def getConstFilesSim(self):
    return [tff.simFile for tfn,tff in self.tensors.items() if tff.isConst]

# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Neural network scheduler - a neural network operation
#

from nnsh_cost import NnshCost
import numpy as np

class NnshOp(object):
  
  def __init__(self, name, attrs):
    self.name = name
    self.attrs = attrs
    self.op = self.attrs["layer_type"]
  
  def getOfmapShape(self):
    return self.attrs['k_attrs']["ofmap_shape"]
  
  def getNpDtype(self):
    t = self.attrs['k_attrs']["data_type"]
    npType = eval("np." + t)
    return npType
  
  # For folded constants
  def getConstNpDtype(self):
    t = self.attrs['k_const']["data_type"]
    npType = eval("np." + t)
    return npType
  
  # Count this op's OFMAP size as state (IFMAPS)
  def getStateCost(self):
    # OFMAP is the state
    ofmapShape = self.getOfmapShape()
    bytesOf = np.empty(ofmapShape, dtype=self.getNpDtype()).nbytes
    c = NnshCost(self.op, self.name, 0, bytesOf, 0)
    return c

  # Count execution of this op; IFMAP size is counted in state (not here)
  def getOpCost(self):
    # OFMAP is computed during execution
    ofmapShape = self.getOfmapShape()
    bytesOf = np.empty(ofmapShape, dtype=self.getNpDtype()).nbytes
    # Bias, etc
    bytesW = 0
    const = self.attrs.get('k_const', None)
    if const:
      kernelShape = const['ofmap_shape']
      bytesW = np.empty(kernelShape, dtype=self.getConstNpDtype()).nbytes
    c = NnshCost(self.op, self.name, bytesW, 0, bytesOf)
    return c
    

class NnshInput(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  

class NnshConst(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)


class NnshConv(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  def getOpCost(self):
    c = super().getOpCost()
    kernelShape = self.attrs['k_attrs']["kernel_shape"]
    bytesW = np.empty(kernelShape, dtype=self.getNpDtype()).nbytes
    cw = NnshCost(self.op, self.name, bytesW, 0, 0)
    c.add(cw)
    return c
  
class NnshMatMul(NnshConv):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)


class NnshBiasAdd(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  
class NnshRelu(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  
class NnshResAdd(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  
class NnshMaxPool(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  
class NnshAvgPool(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  
class NnshReshape(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  
class NnshSoftmax(NnshOp):
  def __init__(self, name, attrs):
    super().__init__(name, attrs)
  


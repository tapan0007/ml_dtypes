# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Neural network scheduler - cost model of one operation
#

class NnshCost(object):
  
  def __init__(self, op, name, w, i, o):
    self.op = op
    self.name = name
    self.bytesW = w
    self.bytesIf = i
    self.bytesOf = o
  
  def __str__(self):
    return "%s w%.3f i%.3f o%.3f  %s" % (self.op,
      self.bytesW / 2**20, self.bytesIf / 2**20, self.bytesOf / 2**20,
      self.name)
  
  def add(self, cost):
    self.bytesW += cost.bytesW
    self.bytesIf += cost.bytesIf
    self.bytesOf += cost.bytesOf

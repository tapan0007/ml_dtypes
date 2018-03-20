# Copyright (C) 2018, Amazon.com. All Rights Reserved
#
# Neural network scheduler
#

import json
from nnsh_graph import NnshGraph
from nnsh_prof import NnshProfile

class NnshOpt(object):

  def __init__(self, jsonFile, verbose, debug):
    self.jsonFile = jsonFile
    self.verbose = verbose
    self.debug = debug
    with open(self.jsonFile) as fh:
      self.jsonData = json.load(fh)
      self.graphJson = NnshGraph(debug, self.jsonData)
      if self.debug > 0:
        self.graphJson.render("debug_graph_json")

  def profile(self, profileType, csvOutput):
    g = None
    if profileType == "folded":
      g = self.graphFolded
    if g:
      prof = NnshProfile()
      cuts = g.getCuts()
      for cut in cuts:
        opCosts = g.getCutCosts(cut)
        prof.addStep(opCosts)
      prof.writeCsv(csvOutput)

  def fold(self):
    foldOpSet = set()
    foldOpSet.add('Const')
    self.graphFolded = self.graphJson.copy()
    self.graphFolded.fold(foldOpSet)
    if self.debug > 0:
      self.graphFolded.render("debug_graph_folded")
  
  def fuse(self):
    fuseRules = {'ConvBiasRelu' : ['Conv', 'BiasAdd', 'Relu']}
    self.graphFused = self.graphFolded.copy()
    self.graphFused.fuse(fuseRules)
    if self.debug > 0:
      self.graphFused.render("debug_graph_fused")
  
  def partition(self):
    pass
  
  def schedule(self):
    pass
  
  def write(self, outFile):
    pass
  

  
  


from utils.consts   import *
from utils.fmapdesc import  OfmapDesc
from layer          import Layer
from convlayer      import ConvLayer
import nets.network

class FullLayer(ConvLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, numOuts):
        assert(isinstance(prev_layer, Layer))
        super(FullLayer, self).__init__(param, prev_layer, numOuts, stride=1, kernel=1)


    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("Full " + baseLayer)

    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Full"

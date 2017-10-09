from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from layer          import Layer
from onetoonelayer  import OneToOneLayer
import nets.network


class SoftMaxLayer(OneToOneLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))
        super(SoftMaxLayer, self).__init__(param, prev_layer)

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("SoftMax " + baseLayer + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_SOFTMAX

    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "SoftMax"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False


from utils.consts    import *
from utils.fmapdesc  import  OfmapDesc
from layer           import Layer
from onetoonelayer   import OneToOneLayer
import nets.network

##########################################################
class BatchNormLayer(OneToOneLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))

        super(BatchNormLayer, self).__init__(param, prev_layer)

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("BNorm " + baseLayer + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_BATCH_NORM

    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "BNorm"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False


from utils.consts    import *
#from utils.fmapdesc  import  IfmapDesc
from utils.fmapdesc  import  OfmapDesc
from layer           import Layer
from onetoonelayer   import OneToOneLayer
import nets.network

##########################################################
class BatchNormLayer(OneToOneLayer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))

        super(BatchNormLayer, self).__init__(ntwk, prev_layer)

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("BNorm " + baseLayer + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_BATCH_NORM

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False


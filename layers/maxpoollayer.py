from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from poollayer       import PoolLayer
import nets.network

##########################################################
class MaxPoolLayer(PoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer, stride, kernel):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))

        super(MaxPoolLayer, self).__init__(ntwk, prev_layer,
            stride=stride, kernel=kernel)

        #self.m_PoolType = poolType

    #-----------------------------------------------------------------
    def __str__(self):
        return self.gPoolLayerStr(self.gLayerType())

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        assert(self.gPrevLayer(0).gOfmapSize()/self.gStride() == self.gOfmapSize())


    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_MAX_POOL

    #-----------------------------------------------------------------
    def gName(self):
        t = "MaxPool"
        return t


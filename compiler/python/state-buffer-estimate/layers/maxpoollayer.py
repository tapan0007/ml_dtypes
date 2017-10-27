from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
from .poollayer       import PoolLayer
import nets.network

##########################################################
class MaxPoolLayer(PoolLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, stride, kernel):
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer,
            stride=stride, kernel=kernel)

        #self.m_PoolType = poolType

    #-----------------------------------------------------------------
    def __str__(self):
        return self.gPoolLayerStr()

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        assert(self.gPrevLayer(0).gOfmapSize() // self.gStride() == self.gOfmapSize())


    #-----------------------------------------------------------------
    def gTypeStr(self):
        t = "MaxPool"
        return t

    #-----------------------------------------------------------------
    def qMaxPoolLayer(self):
        return True


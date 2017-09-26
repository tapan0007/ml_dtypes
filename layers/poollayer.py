from utils.consts    import *
#from utils.fmapdesc  import IfmapDesc
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from subsamplelayer import SubSampleLayer
import nets.network

##########################################################
class PoolLayer(SubSampleLayer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer, stride, kernel, poolType):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))

        super(PoolLayer, self).__init__(ntwk, prev_layer,
            num_ofmaps=None, stride=stride, kernel=kernel)

        self.m_PoolType = poolType

    #-----------------------------------------------------------------
    def __str__(self):
        ks = str(self.gKernel())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        pt = self.gPoolType()
        if pt == POOL_TYPE_MAX:
            t = "MaxPool"
        elif pt == POOL_TYPE_AVG:
            t = "AvgPool"
        else:
            assert(False)
        return (t + baseLayer
                + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                + self.gStateSizesStr())


    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        assert(self.gPrevLayer(0).gOfmapSize()/self.gStride() == self.gOfmapSize())
        

    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_POOL

    def gName(self):
        pt = self.gPoolType()
        if pt == POOL_TYPE_MAX:
            t = "MaxPool"
        elif pt == POOL_TYPE_AVG:
            t = "AvgPool"
        else:
            assert(False)

        return t


    #-----------------------------------------------------------------
    def gPoolType(self):
        return self.m_PoolType

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False


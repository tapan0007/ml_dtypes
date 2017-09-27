from abc             import ABCMeta, abstractmethod

from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from subsamplelayer import SubSampleLayer
import nets.network

##########################################################
class PoolLayer(SubSampleLayer):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer, stride, kernel):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))

        super(PoolLayer, self).__init__(ntwk, prev_layer,
            num_ofmaps=None, stride=stride, kernel=kernel)


    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        assert(self.gPrevLayer(0).gOfmapSize()/self.gStride() == self.gOfmapSize())


    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gSingleBatchInputStateSize(self, batch=1):
        return 0

    #-----------------------------------------------------------------
    def gSingleBatchOutputStateSize(self, batch=1):
        num_ofmaps = self.gNumOfmaps()
        ofmap_size = self.gOfmapSize()
        return ofmap_size * ofmap_size * num_ofmaps

    #-----------------------------------------------------------------
    def gPoolLayerStr(self, typ):
        ks = str(self.gKernel())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        if typ == LAYER_TYPE_MAX_POOL:
            t = "MaxPool"
        elif typ == LAYER_TYPE_AVG_POOL:
            t = "AvgPool"
        else:
            assert(False)

        return (t + baseLayer
                + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                + self.gStateSizesStr())


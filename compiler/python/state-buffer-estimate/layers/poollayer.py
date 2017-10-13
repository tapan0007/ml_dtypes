from abc             import ABCMeta, abstractmethod

from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
from subsamplelayer import SubSampleLayer
import nets.network

##########################################################
class PoolLayer(SubSampleLayer): # abstract class
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, stride, kernel):
        assert(isinstance(prev_layer, Layer))

        super(PoolLayer, self).__init__(param, prev_layer,
            num_ofmaps=None, stride=stride, kernel=kernel)


    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        assert(self.gPrevLayer(0).gOfmapSize()/self.gStride() == self.gOfmapSize())


    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gPoolLayerStr(self, typ):
        ks = str(self.gKernel())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        if typ == LAYER_TYPE_MAX_POOL:
            t = "{MaxPool}"
        elif typ == LAYER_TYPE_AVG_POOL:
            t = "{AvgPool}"
        else:
            assert(False)

        return (self.gName() # + t 
                + baseLayer
                + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                + self.gStateSizesStr())


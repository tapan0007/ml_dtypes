from abc             import ABCMeta, abstractmethod

from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
from .subsamplelayer import SubSampleLayer
import nets.network

##########################################################
class PoolLayer(SubSampleLayer, metaclass = ABCMeta): # abstract class

    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, stride, kernel):
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer,
            num_ofmaps=None, stride=stride, kernel=kernel)


    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gNumPrevLayers() == 1)
        super().verify()


    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gPoolLayerStr(self):
        ks = str(self.gKernel())
        ss = str(self.gStride())
        baseLayer = self.gBaseLayerStr()
        t = "{" + self.gTypeStr() + "}"

        return (self.gName() # + t
                + baseLayer
                + ", kernel=" + ks + "x" + ks + ", stride=" + ss
                + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def qPoolLayer(self):
        return True


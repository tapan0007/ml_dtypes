from utils.consts    import *
from utils.fmapdesc  import  OfmapDesc
from .layer           import Layer
from .onetoonelayer   import OneToOneLayer
import nets.network

##########################################################
class BatchNormLayer(OneToOneLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer)

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("BNorm " + baseLayer + self.gStateSizesStr())

    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(self):
        return "BNorm"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qBatchNormLayer(self):
        return True


from utils.consts    import *
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
    def gName(self):
        return "BNorm"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def gSingleBatchInputStateSize(self, batch=1):
        sz = 0
        for prevLayer in self.gPrevLayers():
            num_ofmaps = prevLayer.gNumOfmaps()
            ofmap_size = prevLayer.gOfmapSize()
            sz += ofmap_size * ofmap_size * num_ofmaps
        return sz

    #-----------------------------------------------------------------
    def gSingleBatchOutputStateSize(self, batch=1):
        num_ofmaps = self.gNumOfmaps()
        ofmap_size = self.gOfmapSize()
        return ofmap_size * ofmap_size * num_ofmaps


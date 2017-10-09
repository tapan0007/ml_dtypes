from utils.consts  import  *
from layer         import Layer
import nets.network

##########################################################
class OneToOneLayer(Layer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, (layerName, batch, ntwk), prev_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        ofmap_desc = prev_layer.gOfmapDesc()
        super(OneToOneLayer, self).__init__((layerName, batch, ntwk), ofmap_desc, (prev_layer,))
        assert(prev_layer.gRawOutputStateSize() == self.gRawOutputStateSize())
        assert(self.gRawInputStateSize() == self.gRawOutputStateSize())

    #-----------------------------------------------------------------
    def verify(self):
        assert(1 == self.gNumPrevLayers())
        prev_layer = self.gPrevLayer(0)
        assert(prev_layer.gOfmapDesc() == self.gOfmapDesc())

    #-----------------------------------------------------------------
    def gBatchInputStateSize(self, batch=1):
        return self.gRawInputStateSize()

    #-----------------------------------------------------------------
    def gBatchOutputStateSize(self, batch=1):
        return self.gRawOutputStateSize()

    #-----------------------------------------------------------------


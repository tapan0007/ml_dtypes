from utils.consts  import  *
from layer         import Layer
import nets.network

##########################################################
class OneToOneLayer(Layer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        ofmap_desc = prev_layer.gOfmapDesc()
        super(OneToOneLayer, self).__init__(ntwk, (prev_layer,), ofmap_desc)
        assert(prev_layer.gRawOutputSize() == self.gRawOutputSize())
        assert(self.gRawInputSize() == self.gRawOutputSize())

    #-----------------------------------------------------------------
    def verify(self):
        assert(1 == self.gNumPrevLayers())
        prev_layer = self.gPrevLayer(0)
        assert(prev_layer.gOfmapDesc() == self.gOfmapDesc())

    #-----------------------------------------------------------------
    def gSingleBatchInputStateSize(self, batch=1):
        return self.gRawInputSize()

    #-----------------------------------------------------------------
    def gSingleBatchOutputStateSize(self, batch=1):
        return self.gRawOutputSize()

    #-----------------------------------------------------------------


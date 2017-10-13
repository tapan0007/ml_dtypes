from utils.consts  import  *
from layer         import Layer
import nets.network

##########################################################
class OneToOneLayer(Layer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))
        ofmap_desc = prev_layer.gOfmapDesc()
        super(OneToOneLayer, self).__init__(param, ofmap_desc, (prev_layer,))
        assert(prev_layer.gRawOutputStateSizeOneBatch() == self.gRawOutputStateSizeOneBatch())

    #-----------------------------------------------------------------
    def verify(self):
        assert(1 == self.gNumPrevLayers())
        prev_layer = self.gPrevLayer(0)
        assert(prev_layer.gOfmapDesc() == self.gOfmapDesc())

    #-----------------------------------------------------------------


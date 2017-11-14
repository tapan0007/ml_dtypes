from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
import nets.network

##########################################################
class CombineLayer(Layer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer, earlier_layer, num_ofmaps):
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapWidth() == earlier_layer.gOfmapWidth())
        assert(prev_layer.gOfmapHeight() == earlier_layer.gOfmapHeight())

        ofmap_desc = OfmapDesc(num_ofmaps, 
                          (prev_layer.gOfmapWidth(), prev_layer.gOfmapHeight()))

        super().__init__(param, ofmap_desc, (prev_layer, earlier_layer))

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qCombineLayer(self):
        return True



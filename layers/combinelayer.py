from utils.consts    import *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
import nets.network

##########################################################
class CombineLayer(Layer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, layerName, ntwk, prev_layer, earlier_layer, num_ofmaps):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        assert(isinstance(earlier_layer, Layer))
        assert(prev_layer.gOfmapSize() == earlier_layer.gOfmapSize())

        ofmap_desc = OfmapDesc(num_ofmaps, prev_layer.gOfmapSize())

        super(CombineLayer, self).__init__(layerName, ntwk, (prev_layer, earlier_layer), ofmap_desc)

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False



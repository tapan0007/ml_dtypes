from utils.consts    import  *
#from utils.fmapdesc  import IfmapDesc
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
import nets.network

##########################################################
class DataLayer(Layer):
    #-----------------------------------------------------------------
    def __init__(self, ntwk, ofmap_desc):
        assert(isinstance(ntwk, nets.network.Network))
        super(DataLayer, self).__init__(ntwk, (), ofmap_desc)

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("Data " + baseLayer
               + self.gStateSizesStr())

    def gName(self):
        return "Data"

    #-----------------------------------------------------------------
    def verify(self):
        return

    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_DATA

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    def qDataLayer(self):
        return True



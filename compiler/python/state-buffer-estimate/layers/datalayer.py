from utils.consts    import  *
from utils.fmapdesc  import OfmapDesc
from layer           import Layer
import nets.network

##########################################################
class DataLayer(Layer):
    #-----------------------------------------------------------------
    def __init__(self, param, ofmap_desc):
        super(DataLayer, self).__init__(param, ofmap_desc, ())

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("Data " + baseLayer
               + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def gTypeStr(self):
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

    #-----------------------------------------------------------------
    def qDataLayer(self):
        return True

    #-----------------------------------------------------------------
    def gBatchInputStateSize(self, batch=1):
        return 0

    #-----------------------------------------------------------------
    def gBatchOutputStateSize(self, batch=1):
        num_ofmaps = self.gNumOfmaps()
        ofmap_size = self.gOfmapSize()
        return ofmap_size * ofmap_size * num_ofmaps

    #-----------------------------------------------------------------
    def qStoreInSB(self): ## override
        return True


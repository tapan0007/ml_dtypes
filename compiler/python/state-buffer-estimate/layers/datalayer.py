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
        return (self.gName() + baseLayer
               + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Data"

    #-----------------------------------------------------------------
    def verify(self):
        return

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qDataLayer(self):
        return True

    #-----------------------------------------------------------------
    def qStoreInSB(self): ## override
        return True

    #-----------------------------------------------------------------
    def qDataLayer(self):
        return True


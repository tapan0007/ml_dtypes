from utils.consts    import  *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
import nets.network

##########################################################
class DataLayer(Layer):
    #-----------------------------------------------------------------
    # TODO: remove default values for input data file name and tensor dimension meaning string
    def __init__(self, param, ofmap_desc, inputDataFileName, dataTensorDimSemantics):
        super().__init__(param, ofmap_desc, ())
        self.__InputDataFileName = inputDataFileName
        self.__DataTensorDimSemantics = dataTensorDimSemantics

    #-----------------------------------------------------------------
    def gJson(self):
        x = {
            "name"          : self.gName(),
            "number_ifmaps" : self.gNumOfmaps(),
            "ifmap_width"   : self.gOfmapWidth(),
            "ifmap_height"  : self.gOfmapHeight(),
            "input_file"    : self.__InputDataFileName,
            "input_dims"    : self.__DataTensorDimSemantics
        }
        return x

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return (self.gName() + baseLayer
               + self.gStateSizesStr())

    #-----------------------------------------------------------------
    def gInputDataFileName(self):
        return self.__InputDataFileName

    #-----------------------------------------------------------------
    def gDataTensorDimSemantics(self):
        return self.__DataTensorDimSemantics

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


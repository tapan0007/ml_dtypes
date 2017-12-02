from utils.consts    import  *
from utils.fmapdesc  import OfmapDesc
from .layer           import Layer
import nets.network

##########################################################
class DataLayer(Layer):
    input_dims_key = "ofmap_format"

    #-----------------------------------------------------------------
    # TODO: remove default values for input data file name and tensor dimension meaning string
    def __init__(self, param, ofmap_desc, inputDataFileName, dataTensorDimSemantics):
        super().__init__(param, ofmap_desc, ())
        self.__InputDataFileName = inputDataFileName
        self.__DataTensorDimSemantics = dataTensorDimSemantics

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        y = {
            Layer.ref_file_key        : self.__InputDataFileName,
            DataLayer.input_dims_key  : self.__DataTensorDimSemantics
        }
        r = self.combineJson( (x, y) )
        return r

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(cls, layerDict, nn):
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        layerName = Layer.gLayerNameFromJson(layerDict)

        inputFileName = layerDict[Layer.ref_file_key]
        tensorSemantics = layerDict[DataLayer.input_dims_key]
        batch = 1

        param = Layer.Param(layerName, batch, nn)
        layer = DataLayer(param, ofmapDesc, inputFileName, tensorSemantics)
        return layer

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
    @classmethod
    def gTypeStr(cls):
        return "Input"

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


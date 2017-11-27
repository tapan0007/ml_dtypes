from utils.consts    import *
from utils.fmapdesc  import  OfmapDesc
from .layer           import Layer
from .onetoonelayer   import OneToOneLayer
import nets.network

##########################################################
class BatchNormLayer(OneToOneLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))

        super().__init__(param, prev_layer)

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(klass, layerDict, nn):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        batch = 1
        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
        layer = BatchNormLayer(param, prevLayers[0])
        return layer

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("BNorm " + baseLayer + self.gStateSizesStr())

    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(klass):
        return "BNorm"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qBatchNormLayer(self):
        return True


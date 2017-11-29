from utils.consts    import *
from utils.fmapdesc import  OfmapDesc
from .layer          import Layer
from .onetoonelayer  import OneToOneLayer
import nets.network


class SoftMaxLayer(OneToOneLayer):
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
    def constructFromJson(cls, layerDict, ntwk):
        batch = 1
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, ntwk)
        param = Layer.Param(layerName, batch, ntwk)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, ntwk)
        assert isinstance(prevLayers, list) and len(prevLayers)==1

        return SoftMaxLayer(param, prevLayers[0])

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("SoftMax " + baseLayer + self.gStateSizesStr())

    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(cls):
        return "SoftMax"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qSoftMaxLayer(self):
        return True



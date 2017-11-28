from abc             import ABCMeta, abstractmethod

from utils.consts  import  *
from .layer         import Layer
from .activlayer    import ActivLayer
import nets.network

##########################################################
class ReluLayer(ActivLayer):
    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))
        super().__init__(param , prev_layer)

    #-----------------------------------------------------------------
    def gJson(self):
        x = super().gJson()
        return x

    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(cls, layerDict, nn):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        batch = 1
        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
        layer = ReluLayer(param, prevLayers[0])
        return layer


    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("Relu " + baseLayer + self.gStateSizesStr())


    #-----------------------------------------------------------------
    @classmethod
    def gTypeStr(cls):
        return "Relu"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qReluLayer(self):
        return True




from abc             import ABCMeta, abstractmethod

from utils.consts   import  *
from .layer          import Layer
from .onetoonelayer  import OneToOneLayer
import nets.network

##########################################################
class ActivLayer(OneToOneLayer, metaclass = ABCMeta): # abstract class

    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))
        super().__init__(param, prev_layer)

    #-----------------------------------------------------------------
    def qActivLayer(self):
        return True


    #-----------------------------------------------------------------
    @classmethod
    def constructFromJson(cls, layerDict, nn):
        layerName = Layer.gLayerNameFromJson(layerDict)
        ofmapDesc = Layer.gOfmapDescFromJson(layerDict, nn)
        batch = 1
        param = Layer.Param(layerName, batch, nn)
        prevLayers = Layer.gPrevLayersFromJson(layerDict, nn)
        assert isinstance(prevLayers, list) and len(prevLayers)==1
        layer = cls(param, prevLayers[0])
        return layer



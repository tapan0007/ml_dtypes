from abc             import ABCMeta, abstractmethod

from utils.consts  import  *
from layer         import Layer
from activlayer    import ActivLayer
import nets.network

##########################################################
class ReluLayer(ActivLayer):
    #-----------------------------------------------------------------
    def __init__(self, layerName, ntwk, prev_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        super(ReluLayer, self).__init__(layerName, ntwk, prev_layer)

    #-----------------------------------------------------------------
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("Relu " + baseLayer + self.gStateSizesStr())


    #-----------------------------------------------------------------
    def gLayerType(self):
        return LAYER_TYPE_ReLU

    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Relu"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False




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
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return ("Relu " + baseLayer + self.gStateSizesStr())


    #-----------------------------------------------------------------
    def gTypeStr(self):
        return "Relu"

    #-----------------------------------------------------------------
    def qPassThrough(self):
        return False

    #-----------------------------------------------------------------
    def qReluLayer(self):
        return True




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
    def __str__(self):
        baseLayer = self.gBaseLayerStr()
        return (self.gTypeStr() + baseLayer + self.gStateSizesStr())


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




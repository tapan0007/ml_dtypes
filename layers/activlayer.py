from utils.consts   import  *
from layer          import Layer
from onetoonelayer  import OneToOneLayer
import nets.network

##########################################################
class ActivLayer(OneToOneLayer): # abstract class
    #-----------------------------------------------------------------
    def __init__(self, ntwk, prev_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        super(ActivLayer, self).__init__(ntwk, prev_layer)



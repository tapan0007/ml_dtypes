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



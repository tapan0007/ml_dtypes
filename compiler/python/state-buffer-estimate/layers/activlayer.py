from abc             import ABCMeta, abstractmethod

from utils.consts   import  *
from layer          import Layer
from onetoonelayer  import OneToOneLayer
import nets.network

##########################################################
class ActivLayer(OneToOneLayer): # abstract class
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, param, prev_layer):
        assert(isinstance(prev_layer, Layer))
        super(ActivLayer, self).__init__(param, prev_layer)


    #-----------------------------------------------------------------
    def gBatchInputStateSize(self):
        return 0

    #-----------------------------------------------------------------
    def gBatchOutputStateSize(self):
        nextSchedLayer = self.gNextSchedLayer()
        if not nextSchedLayer or self.gNextSchedLayer().qConvLayer():
            return self.gRawOutputStateSize()
        else:
            return 0


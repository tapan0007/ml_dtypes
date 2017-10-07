from abc             import ABCMeta, abstractmethod

from utils.consts   import  *
from layer          import Layer
from onetoonelayer  import OneToOneLayer
import nets.network

##########################################################
class ActivLayer(OneToOneLayer): # abstract class
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, layerName, ntwk, prev_layer):
        assert(isinstance(ntwk, nets.network.Network))
        assert(isinstance(prev_layer, Layer))
        super(ActivLayer, self).__init__(layerName, ntwk, prev_layer)


    #-----------------------------------------------------------------
    def gSingleBatchInputStateSize(self, batch=1):
        return 0 

    #-----------------------------------------------------------------
    def gSingleBatchOutputStateSize(self, batch=1):
        nextSchedLayer = self.gNextSchedLayer()
        if not nextSchedLayer or self.gNextSchedLayer().qConvLayer():
            return self.gRawOutputStateSize(batch)
        else:
            return 0


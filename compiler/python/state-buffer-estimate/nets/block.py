from abc             import ABCMeta, abstractmethod

from utils.consts           import  *
import nets.network

##########################################################
class Block(object):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, ntwk):
        assert(isinstance(ntwk, nets.network.Network))
        self.m_network   = ntwk

    #-----------------------------------------------------------------
    @abstractmethod
    def gLastLayer(self):
        return self.m_LastLayer



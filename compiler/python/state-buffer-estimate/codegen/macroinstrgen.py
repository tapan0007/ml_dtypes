
from nets.network import Network
from arch.arch import Arch

import layers.convlayer 
import layers.maxpoollayer
import layers.avgpoollayer

from .genconvlayer    import GenConvLayer
from .genmaxpoollayer import GenMaxPoolLayer
from .genavgpoollayer import GenAvgPoolLayer

##########################################################
class MacroInstrGen(object):
    #-----------------------------------------------------------------
    def createGenMap(self):
        self.__Map = {
            layers.convlayer.ConvLayer : GenConvLayer(self),
            layers.maxpoollayer.MaxPoolLayer : GenMaxPoolLayer(self),
            layers.avgpoollayer.AvgPoolLayer : GenAvgPoolLayer(self),
        }

    #-----------------------------------------------------------------
    def __init__(self, ntwk, arch):
        self.__Network = ntwk
        self.__Arch = arch

        self.createGenMap()

    #-----------------------------------------------------------------
    def gGenFunc(self, layer):
        try:
            generator = self.__Map[layer.__class__]
        except KeyError:
            generator = None
        return generator

    #-----------------------------------------------------------------
    def generate(self):
        for layer in self.__Network.gSchedLayers():
            generator = self.gGenFunc(layer)
            if generator:
                generator.generate(layer)


from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
##from utils.funcs     import kstr
import layers.layer
from schedule.scheduler      import Scheduler

##########################################################
class Network(object):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    class SchedLayerForwRevIter(object):
        def __init__(self, startLayer, forw):
            self.__Forw = forw
            self.__CurrLayer = startLayer

        def __iter__(self):
            return self

        def next(self):
            currLayer = self.__CurrLayer
            if not currLayer:
                raise StopIteration()

            if self.__Forw:
                nextLayer = currLayer.gNextSchedLayer()
            else:
                nextLayer = currLayer.gPrevSchedLayer()

            self.__CurrLayer = nextLayer
            return currLayer

    #-----------------------------------------------------------------
    def __init__(self):
        self.__Layers = [ ]
        self.__LayerNumMajor = -1
        self.__LayerNumMinor = 0
        self.__Levels = None
        self.__CurrLayerId = 0
        self.__DoBatching = False

    #-----------------------------------------------------------------
    def qDoBatching(self):
        return self.__DoBatching

    #-----------------------------------------------------------------
    def rDoBatching(self, batch):
        self.__DoBatching = batch

    #-----------------------------------------------------------------
    def gLevels(self):
        return self.__Levels

    def rLevels(self, levels):
        self.__Levels = levels

    #-----------------------------------------------------------------
    def gLayers(self):
        return self.__Layers

    #-----------------------------------------------------------------
    def gLayer(self, idx):
        return self.__Layers[idx]

    #-----------------------------------------------------------------
    def addLayer(self, layer):
        assert(layer)
        assert( isinstance(layer, layers.layer.Layer) )
        layer.rLayerId(self.__CurrLayerId); self.__CurrLayerId += 1
        self.__Layers.append(layer)
        if layer.qDataLayer() or layer.qConvLayer() or layer.qFullLayer():
            self.__LayerNumMajor += 1
            self.__LayerNumMinor = 0
            numStr = str(self.__LayerNumMajor)
        else:
            numStr = str(self.__LayerNumMajor) + "." + str(self.__LayerNumMinor)
            self.__LayerNumMinor += 1
        layer.rNumberStr(numStr)

    #-----------------------------------------------------------------
    def gNumberLayers(self):
        return len(self.__Layers)

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.gLayer(0).qDataLayer())
        numLayers = self.gNumberLayers()

        for layer in self.gLayers(): # self.__Layers:
            layer.verify()





    #-----------------------------------------------------------------
    @abstractmethod
    def construct(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def gName(self):
        assert(False)

    #-----------------------------------------------------------------
    def gSchedLayers(self):
        #-------------------------------------------------------------
        return Network.SchedLayerForwRevIter(self.__Layers[0], True)

    #-----------------------------------------------------------------
    def gReverseSchedLayers(self):
        return Network.SchedLayerForwRevIter(self.__Layers[-1], False)






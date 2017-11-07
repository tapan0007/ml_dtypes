from utils.consts    import  *
import layers.layer
#from layers.layer import Layer

##########################################################
class LayerLevel(object):
    #-----------------------------------------------------------------
    def __init__(self, levelNum, initLayers):
        isinstance(initLayers, list)
        self.__LevelNum = levelNum
        self.__Layers = list(initLayers)

    #-----------------------------------------------------------------
    def remove(self, layer):
        assert(self.qContainsLayer(layer))
        self.__Layers.remove(layer)

    #-----------------------------------------------------------------
    def append(self, layer):
        assert(isinstance(layer, layers.layer.Layer))
        self.__Layers.append(layer)

    #-----------------------------------------------------------------
    def gLevelNum(self):
        return self.__LevelNum

    #-----------------------------------------------------------------
    def gLayers(self):
        return iter(self.__Layers)

    #-----------------------------------------------------------------
    def qContainsLayer(self, layer):
        return layer in self.__Layers

    #-----------------------------------------------------------------
    def gNumberLayers(self):
        return len(self.__Layers)

    #-----------------------------------------------------------------
    def qDataLevel(self):
        for layer in self.__Layers:
            if not layer.qDataLayer():
                return False
        return True

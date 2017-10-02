from utils.consts    import  *
import layers.layer
#from layers.layer import Layer

##########################################################
class LayerLevel(object):
    #-----------------------------------------------------------------
    def __init__(self, levelNum, initLayers):
        isinstance(initLayers, list)
        assert(initLayers==[] or isinstance(initLayers[0], layers.layer.Layer))
        self.__LevelNum = levelNum
        self.__Layers = initLayers

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
    def gNumberLayers(self):
        return len(self.__Layers)

    #-----------------------------------------------------------------
    def qDataLevel(self):
        for layer in self.__Layers:
            if layer.gLayerType() != LAYER_TYPE_DATA:
                return False
        return True


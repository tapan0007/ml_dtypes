from utils.consts    import  *
import layers.layer
from schedule.layerlevel import LayerLevel


##########################################################
class Scheduler(object):
    #-----------------------------------------------------------------
    def __init__(self):
        self.__Layers = layers
        self.__Levels = None

    #-----------------------------------------------------------------
    def __gNumberLayers(self):
        return len(self.__Layers)

    #-----------------------------------------------------------------
    def gLevels(self):
        return self.__Levels

    #-----------------------------------------------------------------
    #def rLevels(self, levels):
    #    self.__Levels = levels

    #-----------------------------------------------------------------
    # Level[i] = layers without predecessors in: All-Layers - Union{k : k in [0,i) : Level[k]}
    def __levelize(self):
        for layer in self.__Layers:
            layer.m_NumPredecessors = layer.gNumPrevLayers()
            layer.rEarlyLevel(-1)
        Levels = []

        # get layers without predecessors

        currLevelNum = 0; assert(currLevelNum == len(Levels))
        lev0 = filter(lambda lyr : lyr.m_NumPredecessors == 0, self.__Layers)
        currLevel = LayerLevel(currLevelNum, lev0)

        for layer in currLevel.gLayers():
            layer.rEarlyLevel(currLevelNum)

        Levels.append(currLevel)  ## this is level 0
        numUnprocessedLayers = self.__gNumberLayers() - currLevel.gNumberLayers()

        while numUnprocessedLayers > 0:
            nextLevelNum = currLevelNum + 1; assert(nextLevelNum == len(Levels))
            nextLevel = LayerLevel(nextLevelNum, [])
            for currLayer in currLevel.gLayers():
                for nextLayer in currLayer.gNextLayers():
                    nextLayer.m_NumPredecessors -= 1
                    if nextLayer.m_NumPredecessors == 0:  ## all predecessors in previous layers
                        nextLevel.append(nextLayer)

            currLevel = nextLevel; currLevelNum = nextLevelNum
            numUnprocessedLayers -= currLevel.gNumberLayers()
            Levels.append(currLevel)
            for layer in currLevel.gLayers():
                layer.rEarlyLevel(currLevelNum)

        self.__Levels = Levels
        self.__calculateLateLevels()

    #-----------------------------------------------------------------
    def __calculateLateLevels(self):
        lastLevel = len(self.__Levels)

        revLevels = list(self.__Levels)
        revLevels.reverse()
        for level in revLevels:
            for layer in level.gLayers():
                minNextLastLev = lastLevel
                for nextLayer in layer.gNextLayers():
                    minNextLastLev = min(minNextLastLev, nextLayer.gLateLevel())
                layer.rLateLevel(minNextLastLev - 1)

    #-----------------------------------------------------------------
    def schedule(self, layers):

        self.__Layers = layers
        self.__Levels = None

        self.__levelize()
        Levels = self.__Levels
        assert(Levels[0].gNumberLayers() == 1 and Levels[0].qDataLevel())


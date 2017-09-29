from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
from utils.funcs     import kstr
import layers.layer

##########################################################
class Network(object):
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self):
        self.__Layers = [ ]
        self.__LayerNumMajor = -1
        self.__LayerNumMinor = 0

    #-----------------------------------------------------------------
    def gLayer(self, idx):
        return self.g_Layers[idx]

    #-----------------------------------------------------------------
    def addLayer(self, layer):
        assert(layer)
        assert( isinstance(layer, layers.layer.Layer) )
        self.__Layers.append(layer)
        t = layer.gLayerType()
        if (t == LAYER_TYPE_DATA or t == LAYER_TYPE_CONV or t == LAYER_TYPE_FULL):
            self.__LayerNumMajor += 1
            self.__LayerNumMinor = 0
            numStr = str(self.__LayerNumMajor)
        else:
            numStr = str(self.__LayerNumMajor) + "." + str(self.__LayerNumMinor)
            self.__LayerNumMinor += 1
        layer.rNumberStr(numStr)

    #-----------------------------------------------------------------
    def gLayers(self):
        return self.__Layers

    #-----------------------------------------------------------------
    def gNumberLayers(self):
        return len(self.gLayers())

    #-----------------------------------------------------------------
    def verify(self):
        assert(self.__Layers[0].gLayerType() == LAYER_TYPE_DATA)
        numLayers = self.gNumberLayers()

        for layer in self.__Layers:
            layer.verify()



    #-----------------------------------------------------------------
    def printDot(self):
        f1=open(self.gName()+".dot", 'w')

        graphName = self.gName().replace("-", "_").replace(".", "_")
        print >>f1, 'digraph', graphName, "{"

        for layer in self.__Layers:
            print >>f1, '  ', layer.gDotIdLabel()

        print >>f1

        for layer in self.__Layers:
            for nextLayer in layer.gNextLayers():
                print >>f1, '  ', layer.gDotId(), '->', nextLayer.gDotId(), ';'

        print >>f1, '}'
        print >>f1

    #-----------------------------------------------------------------
    def printMe(self):
        prevNl = False
        maxStateSize = 0
        layerNumMajor = 0
        layerNumMinor = 0
        self.m_PrevLayer = None

        for layer in self.__Layers:
            if layer.gDenseBlockStart() >= 0:
                if not prevNl:
                    print
                print ">>> Starting dense block " + str(layer.gDenseBlockStart())
            elif layer.gTranBlockStart() >= 0:
                if not prevNl:
                    print
                print ">>> Starting tran block " + str(layer.gTranBlockStart())

            totalStateSize = layer.gSingleBatchTotalStateSize()
            if totalStateSize > maxStateSize:
                maxStateSize = totalStateSize

            numStr = layer.gNumberStr()
            print (numStr + " " + str(layer))
            layer.m_NumStr = numStr

            prevNl = False
            if layer.gDenseBlockEnd() >= 0:
                print "<<< Ending dense block " + str(layer.gDenseBlockEnd())
                print
                prevNl = True
            elif layer.gTranBlockEnd() >= 0:
                print "<<< Ending tran block " + str(layer.gTranBlockEnd())
                print
                prevNl = True

            self.m_PrevLayer =layer

        print "Max state size =", kstr(maxStateSize)

    #-----------------------------------------------------------------
    # Level[i] = layers without predecessors in: All-Layers - Union{k : k in [0,i) : Level[k]}
    def levelize(self):
        for layer in self.gLayers():
            layer.m_NumPredecessors = layer.gNumPrevLayers()
            layer.rEarlyLevel(-1)
        Levels = []

        # get layers without predecessors
        currLevel = filter(lambda lyr : lyr.m_NumPredecessors == 0, self.gLayers())
        currLevelNum = 0; assert(currLevelNum == len(Levels))
        for layer in currLevel:
            layer.rEarlyLevel(currLevelNum)

        Levels.append(currLevel)  ## this is level 0
        numUnprocessedLayers = self.gNumberLayers() - len(currLevel)

        while numUnprocessedLayers > 0:
            nextLevel = []
            nextLevelNum = currLevelNum + 1; assert(nextLevelNum == len(Levels))
            for currLayer in currLevel:
                for nextLayer in currLayer.gNextLayers():
                    nextLayer.m_NumPredecessors -= 1
                    if nextLayer.m_NumPredecessors == 0:  ## all predecessors in previous layers
                        nextLevel.append(nextLayer)

            currLevel = nextLevel; currLevelNum = nextLevelNum
            numUnprocessedLayers -= len(currLevel)
            Levels.append(currLevel)
            for layer in currLevel:
                layer.rEarlyLevel(currLevelNum)

        self.__Levels = Levels
        self.calculateLateLevels()

    #-----------------------------------------------------------------
    def calculateLateLevels(self):
        lastLevel = len(self.__Levels)

        revLevels = list(self.__Levels)
        revLevels.reverse()
        for level in revLevels:
            for layer in level:
                minNextLastLev = lastLevel
                for nextLayer in layer.gNextLayers():
                    minNextLastLev = min(minNextLastLev, nextLayer.gLateLevel())
                layer.rLateLevel(minNextLastLev - 1)


    #-----------------------------------------------------------------
    def printLevels(self):
        for level in self.__Levels:
            for layer in level:
                print (layer.gNameNum() + '[' + str(layer.gEarlyLevel()) + ',' + str(layer.gLateLevel()) + '] '),
            print


    #-----------------------------------------------------------------
    def schedule(self):
        self.levelize()
        Levels = self.__Levels
        assert(len(Levels[0]) == 1 and Levels[0][0].gLayerType() == LAYER_TYPE_DATA)


    #-----------------------------------------------------------------
    @abstractmethod
    def construct(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def gName(self):
        assert(False)



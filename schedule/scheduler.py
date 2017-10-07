import functools

from utils.consts    import  *
import layers.layer
from schedule.layerlevel import LayerLevel


def breakFunc(n):
    return n + 1

##########################################################
class Scheduler(object):
    #-----------------------------------------------------------------
    def __init__(self):
        self.__Layers = None
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
        self.verifyLevelization()

    #-----------------------------------------------------------------
    def verifyLevelization(self):
        for level in self.__Levels:
            levNum = level.gLevelNum()
            for layer in level.gLayers():
                assert(levNum == layer.gEarlyLevel())
                for nextLayer in layer.gNextLayers():
                    ## cannot say anything about layer.Late and nextLayer.Early
                    assert(layer.gEarlyLevel() < nextLayer.gEarlyLevel())
                    assert(layer.gLateLevel() < nextLayer.gLateLevel())
                    assert(layer.gEarlyLevel() <= layer.gLateLevel())

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
    def schedule(self, ntwk):
        self.__Network = ntwk
        self.__Layers = ntwk.gLayers()
        self.__Levels = None

        self.__levelize()
        Levels = self.__Levels
        assert(Levels[0].gNumberLayers() == 1 and Levels[0].qDataLevel())


        ## Move layers with input smaller than output to latest level for the layer
        for layer in self.__Layers:
            if layer.gEarlyLevel() == layer.gLateLevel():
                continue
            if layer.gRawInputStateSize() < layer.gRawOutputStateSize():
                # move layer to latest level
                breakFunc(2)
                earlyLevel = self.__Levels[layer.gEarlyLevel()]
                lateLevel = self.__Levels[layer.gLateLevel()]

                assert(earlyLevel.qContainsLayer(layer))
                assert(earlyLevel.gLevelNum() == layer.gEarlyLevel())
                assert(lateLevel.gLevelNum() == layer.gLateLevel())

                earlyLevel.remove(layer)
                lateLevel.append(layer)
                assert(not earlyLevel.qContainsLayer(layer))
                assert(lateLevel.qContainsLayer(layer))
            else:
                # keep it early
                pass

        ## Schedule within level
        self.__currSchedule = 0
        for level in self.__Levels:
            self.__scheduleLevel(level)
        self.__linkSchedLayers()
    
        self.calcSbMem()

    #-----------------------------------------------------------------
    def __linkSchedLayers(self):
        for layer in self.__Layers:
            mysch1 = layer.gSchedule() + 1
            for otherLayer in self.__Layers:
                if mysch1 == otherLayer.gSchedule():
                    assert(not layer.gNextSchedLayer())
                    assert(not otherLayer.gPrevSchedLayer())
                    layer.rNextSchedLayer(otherLayer)
                    otherLayer.rPrevSchedLayer(layer)
                    break

        layerWithoutNextSched = None
        layerWithoutPrevSched = None

        for layer in self.__Layers:

            nextSchedLayer = layer.gNextSchedLayer()
            if nextSchedLayer:
                assert(nextSchedLayer.gPrevSchedLayer() == layer)
                assert(layer.gSchedule() + 1 == nextSchedLayer.gSchedule())
            else:
                assert(not layerWithoutNextSched)
                layerWithoutNextSched = layer

            prevSchedLayer = layer.gPrevSchedLayer()
            if prevSchedLayer:
                assert(prevSchedLayer.gNextSchedLayer() == layer)
                assert(prevSchedLayer.gSchedule() + 1 == layer.gSchedule())
            else:
                assert(not layerWithoutPrevSched)
                layerWithoutPrevSched = layer

        assert(layerWithoutNextSched and layerWithoutPrevSched)

    #-----------------------------------------------------------------
    def __scheduleLevel(self, level):
        if level.gNumberLayers() == 1:
            for layer in level.gLayers():
                layer.rSchedule(self.__currSchedule)
                self.__currSchedule += 1
            return

        # Schedule a multi-layer level
        # If two (or more) layers have the same successor, schedule one after another
        # Schedule the layer with smaller out-state-buffer footprint later
        # Rething this for multi-successor layers
        levelCopy = list(level.gLayers())
        self.__sortLayers(levelCopy)
        for layer in level.gLayers():
            layer.rSchedule(self.__currSchedule)
            self.__currSchedule += 1
        return


    #-----------------------------------------------------------------
    def __sortLayers(self, levelCopy):
        #-------------------------------------------------------------
        def compareLayer(layer1, layer2):
            numNext1 = layer1.gNumNextLayers()
            numNext2 = layer2.gNumNextLayers()
            if numNext1 < numNext2:
                return -1
            elif numNext1 > numNext2:
                return 1

            id1 = layer1.gLayerId()
            id2 = layer2.gLayerId()
            if id1 < id2:
                return -1
            elif id1 > id2:
                return 1
            elif layer1.gBatchTotalStateSize() < layer1.gBatchTotalStateSize():
                return -1
            else:
                return 1


        #-------------------------------------------------------------
        sorted(levelCopy, key=functools.cmp_to_key(compareLayer))

    #-----------------------------------------------------------------
    def processLayerSbMem(self, layer):
        if not layer.qStoreInSB():
            return
        outSize = layer.gBatchOutputStateSize()
        self.__CurrMem += outSize
        layer.changeRefCount(layer.gNumNextLayers())
        layer.rTotMem(self.__CurrMem)

        if self.__CurrMem > self.__HighMemWatermark:
            self.__HighMemWatermark = self.__CurrMem 

        for inSbLayer in layer.gPrevSbLayers():
            assert(inSbLayer.qStoreInSB())
            inSbLayer.changeRefCount(-1)  ## decrease ref count by 1
            if inSbLayer.gRefCount() == 0:
                oneInSize = inSbLayer.gBatchOutputStateSize()
                self.__CurrMem -= oneInSize

    #-----------------------------------------------------------------
    def addPrevSbLayers(self, layer):
        for prevLayer in layer.gPrevLayers():
            if prevLayer.qStoreInSB():
                layer.addPrevSbLayer(prevLayer)
            else:
                for sbLayer in prevLayer.gPrevSbLayers():
                    if not sbLayer in layer.gPrevSbLayers():
                        layer.addPrevSbLayer(sbLayer)

    #-----------------------------------------------------------------
    def calcSbMem(self):
        self.__CurrMem = 0
        self.__HighMemWatermark = 0

        for layer in self.__Network.gSchedLayers():
            self.addPrevSbLayers(layer)

        for layer in self.__Network.gSchedLayers():
            self.processLayerSbMem(layer)


    #-----------------------------------------------------------------
    def gCurrMem(self):
        return self.__CurrMem

    #-----------------------------------------------------------------
    def gHighMemWatermark(self):
        return self.__HighMemWatermark


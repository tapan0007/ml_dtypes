import functools

from utils.consts   import  *
import layers.layer
from schedule.layerlevel import LayerLevel


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
    # All layers go to the earliest level:w
    #-----------------------------------------------------------------
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
        for layer in self.__Layers:
            layer.rCurrLevel(layer.gEarlyLevel())
        self.__verifyLevelization()

    #-----------------------------------------------------------------
    def __verifyLevelization(self):
        for level in self.__Levels:
            levNum = level.gLevelNum()
            for layer in level.gLayers():
                assert(levNum == layer.gEarlyLevel())
                for nextLayer in layer.gNextLayers():
                    ## cannot say anything about layer.Late and nextLayer.Early
                    assert(layer.gEarlyLevel() < nextLayer.gEarlyLevel())
                    assert(layer.gLateLevel() < nextLayer.gLateLevel())
                    assert(layer.gEarlyLevel() <= layer.gLateLevel())
                    assert(layer.gEarlyLevel() <= layer.gCurrLevel() and layer.gCurrLevel() <= layer.gLateLevel())

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
    def Schedule(self, ntwk):
        self.__Network = ntwk
        self.__Layers = ntwk.gLayers()
        self.__Levels = None

        ##self.__calcFanoutBatch()

        self.__levelize()
        Levels = self.__Levels
        assert(Levels[0].gNumberLayers() == 1 and Levels[0].qDataLevel())


        ## Move layers with input smaller than output to latest level for the layer
        for layer in self.__Layers:
            if layer.gEarlyLevel() == layer.gLateLevel():
                continue
            earlyLevel = self.__Levels[layer.gEarlyLevel()]
            lateLevel = self.__Levels[layer.gLateLevel()]
            assert(layer.gCurrLevel() == layer.gEarlyLevel())
            assert(earlyLevel.qContainsLayer(layer))
            assert(earlyLevel.gLevelNum() == layer.gEarlyLevel())
            assert(lateLevel.gLevelNum() == layer.gLateLevel())
            assert(not lateLevel.qContainsLayer(layer))

            if layer.gInputSize() < layer.gOutputSize():
                earlyLevel.remove(layer)
                lateLevel.append(layer)
                layer.rCurrLevel(layer.gLateLevel())
                assert(not earlyLevel.qContainsLayer(layer))
                assert(lateLevel.qContainsLayer(layer))

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
            else:
                sz1 = layer1.gRawInputStateSize() + layer1.gRawOutputStateSize() 
                sz2 = layer2.gRawInputStateSize() + layer2.gRawOutputStateSize()
                if sz1 < sz2:
                    return -1
                else:
                    return 1


        #-------------------------------------------------------------
        sorted(levelCopy, key=functools.cmp_to_key(compareLayer))

    #-----------------------------------------------------------------
    # Before processing layer, certain amount of earlier memory is held in SB.
    # The memory equals the layer's inputs plus residue (memory held for future
    # layers).
    #
    # After this layer is processed all its inputs could potentially be released,
    # except those held for the future (residues). The memory after the layer
    # is earlier mem + layer output - inputs that can be released
    #
    # Residue for this layer is all memory that must be held. It equals:
    # 1. Memory held by future layers that is input to this layer
    # 2. Memory held by future layers that is NOT input to this layer
    # So, if memory before layer is inMem = a + b + c:
    # a. input that can be released
    # b. input that cannot be released
    # c. non-input (that cannot be released because it is held by other future layers)
    # Residue = b + c = (a + b + c) - a = inMem - input.that.can.be.released.
    #
    # Total memory after is
    # b + c + layer.out = inMem - input.that.can.be.released + layer.out
    #
    def __processLayerSbMemForResidueWithoutBatching(self, layer):
        assert(layer.qStoreInSB())

        ## all subsequent layers refer to this layer
        layer.changeRefCount(layer.gNumNextLayers()) 
        outSize = layer.gOutputStateMemWithoutBatching()
        inMemBefore = self.gCurrInMem()

        inputCanRelease = 0
        inputCannotRelease = 0
        for inSbLayer in layer.gPrevSbLayers():
            assert(layer != inSbLayer and inSbLayer.qStoreInSB())
            inSbLayer.changeRefCount(-1)  ## decrease ref count by 1
            inSbMem = inSbLayer.gOutputStateMemWithoutBatching()
            if inSbLayer.gRefCount() == 0:
                inputCanRelease += inSbMem
            else:
                inputCannotRelease += inSbMem

        residueMem = inMemBefore - (inputCanRelease)
        inMemAfter = residueMem + outSize

        layer.rResMemWithoutBatching(residueMem)
        self.__CurrInMem = inMemAfter

    #-----------------------------------------------------------------
    def __processLayerSbMemForResidueWithBatching(self, layer):
        assert(layer.qStoreInSB())

        ## all subsequent layers refer to this layer
        layer.changeRefCount(layer.gNumNextLayers()) 
        outSize = layer.gOutputStateMemWithBatching()
        inMemBefore = self.gCurrInMem()

        canRelease = 0
        cannotRelease = 0
        for inSbLayer in layer.gPrevSbLayers():
            assert(layer != inSbLayer and inSbLayer.qStoreInSB())
            inSbLayer.changeRefCount(-1)  ## decrease ref count by 1
            inSbMem = inSbLayer.gOutputStateMemWithBatching()
            if inSbLayer.gRefCount() == 0:
                canRelease += inSbMem
            else:
                cannotRelease += inSbMem

        inMemAfter = inMemBefore + outSize - canRelease
        residueMem = inMemBefore - (canRelease + cannotRelease)

        layer.rResMemWithBatching(residueMem)
        self.__CurrInMem = inMemAfter

    #-----------------------------------------------------------------
    def __calcFanoutBatch(self):
        for layer in self.__Network.gLayers():
            maxFanoutBatchFactor = 0
            numFanouts = 0
            for fanoutLayer in layer.gNextLayers():
                numFanouts += 1
                fob = fanoutLayer.gBatchFactor()
                if fob > maxFanoutBatchFactor:
                    maxFanoutBatch = fob
            assert(numFanouts == 0 or maxFanoutBatch > 0)
            #layer.rMaxFanoutBatchFactor(maxFanoutBatchFactor)


    #-----------------------------------------------------------------
    def __addPrevSbLayers(self, layer):
        for prevLayer in layer.gPrevLayers():
            if prevLayer.qStoreInSB():
                layer.addPrevSbLayer(prevLayer)
            else:
                for sbLayer in prevLayer.gPrevSbLayers():
                    if not sbLayer in layer.gPrevSbLayers():
                        layer.addPrevSbLayer(sbLayer)

    #-----------------------------------------------------------------
    #  Li: Batch Oi, Output size Oi
    #
    #  L1  ---L2---L3----------------\
    #       |                         L6
    #       +------------L4---L5 ----/
    #   L6:   B6*O6
    #   L5:   B5*O5     + (B6-B5)*O5
    #         self mem    L5 batch
    #   L4:   B4*O4     + (B5-B4)*O4 + (B6-B5)*O5
    #         self mem    L4 batch
    #   L3:   B3*O3     + (B6-B3)*O3
    #         self mem    L3 batch
    #   L2:   B2*O2     + (B3-B2)*O2 (B6-B3)*O3
    #         self mem    L2 batch
    #   L1Ba: (B2-B1)*O1 + (B3-B2)*O2
    #         L1 batch a
    #   L1Bb: (B4-B1)*O1 + (B5-B4)*O4
    #         L1 batch b
    #   L1:   B1*O1(self mem) + max(L1a, L1b)
    #-----------------------------------------------------------------
    def __processSbConnectionForBatching(self, prevLayer, nextLayer):
        assert(prevLayer.qStoreInSB() and nextLayer.qStoreInSB())
        deltaBatch = nextLayer.gBatchFactor() - prevLayer.gBatchFactor() 
        assert(deltaBatch >= 0)
        myBatchMem = deltaBatch * prevLayer.gOutputStateMemWithoutBatching()
        batchMem = myBatchMem + nextLayer.gBatchMem()
        if batchMem > prevLayer.gBatchMem():
            prevLayer.rBatchMem(batchMem)

    #-----------------------------------------------------------------
    def calcSbMem(self):
        network = self.__Network

        for layer in network.gSchedLayers():
            self.__addPrevSbLayers(layer)

        # First determine residue without batching
        self.__CurrInMem = 0
        for layer in network.gSchedLayers():
            if not layer.qStoreInSB():
                continue
            self.__processLayerSbMemForResidueWithoutBatching(layer)

        # First determing batching
        for layer in network.gReverseSchedLayers():
            if not layer.qStoreInSB():
                continue
            for inSbLayer in layer.gPrevSbLayers():
                self.__processSbConnectionForBatching(inSbLayer, layer)

        # Second determine residue with batching
        self.__CurrInMem = 0
        for layer in network.gSchedLayers():
            if not layer.qStoreInSB():
                continue
            self.__processLayerSbMemForResidueWithBatching(layer)

    #-----------------------------------------------------------------
    def gCurrInMem(self):
        return self.__CurrInMem

    #-----------------------------------------------------------------
    def gHighMemWatermark(self):
        return self.__HighMemWatermark


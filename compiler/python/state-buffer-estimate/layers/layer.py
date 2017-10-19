from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
from utils.consts    import SCHED_MEM_FORMAT
from utils.fmapdesc  import OfmapDesc
from utils.funcs     import kstr, Kstr
import nets.network

##########################################################
class Layer(object): # abstract class
    __metaclass__ = ABCMeta

    ######################################################
    class Param(object):
        def __init__(self, layerName, batch, ntwk):
            self.__LayerBatchNetwork = (layerName, batch, ntwk)

        def gLayerName(self):
            return self.__LayerBatchNetwork[0]

        def gBatchFactor(self):
            return self.__LayerBatchNetwork[1]

        def gNetwork(self):
            return self.__LayerBatchNetwork[2]

        def gAll(self):
            return self.__LayerBatchNetwork

    #-----------------------------------------------------------------
    def __init__(self, param, ofmap_desc, prev_layers):
        assert(isinstance(param, Layer.Param))
        (layerName, batch, ntwrk) = param.gAll()
        assert(isinstance(layerName, str))
        assert(isinstance(ntwrk, nets.network.Network))
        assert(isinstance(ofmap_desc, OfmapDesc))
        assert(isinstance(prev_layers, tuple))
        for prevLayer in prev_layers:
            assert(isinstance(prevLayer, Layer))
        assert(batch >= 1)

        self.__LayerName        = layerName
        if ntwrk.qDoBatching():
            self.__BatchFactor     = batch
        else:
            self.__BatchFactor     = 1
        self.__BatchMem         = 0

        self.__Network          = ntwrk
        self.__Ofmap_desc       = ofmap_desc.copy()
        self.__Id               = None
        self.__NextSchedLayer   = None
        self.__PrevSchedLayer   = None

        self.__DenseBlockStart  = -1
        self.__DenseBlockEnd    = -1
        self.__TranBlockStart   = -1
        self.__TranBlockEnd     = -1

        self.__NextLayers       = []
        self.__PrevLayers       = []
        self.__PrevSbLayers     = []

        self.__schedule         = None  ## number in [0, NUM_LAYERS-1]
        # counts the number layers that need to be executed
        # and need this layer's
        self.__RefCount         = 0
        self.__ResMemWithBatching = 0
        self.__ResMemWithoutBatching = 0


        assert( (len(prev_layers) == 0) == self.qDataLayer() )
        self.__PrevLayers.extend(prev_layers)
        for prevLayer in prev_layers:
            prevLayer.addNextLayer(self)

        ntwrk.addLayer(self) ## will assign index


    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    def qSubSamleLayer(self):
        return False

    #-----------------------------------------------------------------
    def qConvLayer(self):
        return False

    #-----------------------------------------------------------------
    def qPoolLayer(self):
        return False

    #-----------------------------------------------------------------
    def qMaxPoolLayer(self):
        return False

    #-----------------------------------------------------------------
    def qAvgPoolLayer(self):
        return False

    #-----------------------------------------------------------------
    def qOneToOneLayer(self):
        return False

    #-----------------------------------------------------------------
    def qActivLayer(self):
        return False

    #-----------------------------------------------------------------
    def qReluLayer(self):
        return False

    #-----------------------------------------------------------------
    def qBatchNormLayer(self):
        return False

    #-----------------------------------------------------------------
    def qDataLayer(self):
        return False

    #-----------------------------------------------------------------
    def qCombineLayer(self):
        return False

    #-----------------------------------------------------------------
    def qConcatLayer(self):
        return False

    #-----------------------------------------------------------------
    def qAddLayer(self):
        return False

    #-----------------------------------------------------------------
    def qSoftMaxLayer(self):
        return False

    #-----------------------------------------------------------------
    def qFullLayer(self):
        return False


    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def gTypeStr(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def verify(self):
        assert(False)


    #-----------------------------------------------------------------
    def gBatchFactor(self):
        return self.__BatchFactor

    #-----------------------------------------------------------------
    def gBatchMem(self):
        return self.__BatchMem

    #-----------------------------------------------------------------
    def rBatchMem(self, mem):
        self.__BatchMem = mem


    #-----------------------------------------------------------------
    def gResMemWithoutBatching(self):
        return self.__ResMemWithBatching

    #-----------------------------------------------------------------
    def rResMemWithoutBatching(self, mem):
        self.__ResMemWithBatching = mem

    #-----------------------------------------------------------------
    def gResMemWithBatching(self):
        return self.__ResMemWithBatching

    #-----------------------------------------------------------------
    def rResMemWithBatching(self, mem):
        self.__ResMemWithBatching = mem


    #-----------------------------------------------------------------
    def gInputSize(self):
        sz = 0
        for inLayer in self.gPrevLayers():
            sz += inLayer.gOutputSize()
        return sz

    #-----------------------------------------------------------------
    def gOutputSize(self):
        oneBatchSize = self.gNumOfmaps() * (self.gOfmapSize() * self.gOfmapSize())
        return oneBatchSize

    #-----------------------------------------------------------------
    def gInputStateMemWithoutBatching(self):
        #assert(self.qStoreInSB())
        sz = 0
        for inSbLayer in self.gPrevSbLayers():
            sz += inSbLayer.gOutputStateMemWithoutBatching()
        return sz

    #-----------------------------------------------------------------
    def gOutputStateMemWithoutBatching(self):
        assert(self.qStoreInSB())
        if self.qStoreInSB():
            oneBatchSize = self.gOutputSize()
            return oneBatchSize
        else:
            return 0

    #-----------------------------------------------------------------
    def gOutputStateMemWithBatching(self):
        assert(self.qStoreInSB())
        return self.gBatchFactor() * self.gOutputStateMemWithoutBatching()



    #-----------------------------------------------------------------
    def gLayerId(self):
        return self.__Id

    #-----------------------------------------------------------------
    def rLayerId(self, id):
        self.__Id = id

    #-----------------------------------------------------------------
    def gSchedule(self):
        return self.__schedule

    def rSchedule(self, sch):
        self.__schedule = sch

    #-----------------------------------------------------------------
    def gCurrLevel(self):
        return self.__CurrLevel

    #-----------------------------------------------------------------
    def rCurrLevel(self, lev):
        assert(self.gEarlyLevel() <= lev and lev <= self.gLateLevel())
        self.__CurrLevel = lev

    #-----------------------------------------------------------------
    def gEarlyLevel(self):
        return self.__EarlyLevel

    def rEarlyLevel(self, level):
        self.__EarlyLevel = level

    #-----------------------------------------------------------------
    def gLateLevel(self):
        return self.__LateLevel

    def rLateLevel(self, level):
        self.__LateLevel = level

    #-----------------------------------------------------------------
    def gPrevLayers(self):
        return iter(self.__PrevLayers)

    #-----------------------------------------------------------------
    def gPrevLayer(self, idx):
        assert(0 <= idx and idx < self.gNumPrevLayers())
        return self.__PrevLayers[idx]

    #-----------------------------------------------------------------
    def gNumPrevLayers(self):
        return len(self.__PrevLayers)

    #-----------------------------------------------------------------
    def gNextLayers(self):
        return iter(self.__NextLayers)

    #-----------------------------------------------------------------
    def gNextLayer(self, idx):
        assert(0 <= idx and idx < self.gNumNextLayers())
        return self.__NextLayers[idx]

    #-----------------------------------------------------------------
    def gNumNextLayers(self):
        return len(self.__NextLayers)

    #-----------------------------------------------------------------
    def addNextLayer(self, nextLayer):
        self.__NextLayers.append(nextLayer)

    #-----------------------------------------------------------------
    def gMaxNextLayerNumberWeights(self):
        maxNumWeights = 0
        for nextLayer in self.gNextLayers():
            numWeights = nextLayer.gNumberWeights()
            if numWeights > maxNumWeights:
                maxNumWeights = numWeights
        return maxNumWeights


    #-----------------------------------------------------------------
    def gDenseBlockStart(self):
        return self.__DenseBlockStart

    #-----------------------------------------------------------------
    def rDenseBlockStart(self, val):
        self.__DenseBlockStart = val

    #-----------------------------------------------------------------
    def gDenseBlockEnd(self):
        return self.__DenseBlockEnd

    #-----------------------------------------------------------------
    def rDenseBlockEnd(self, val):
        self.__DenseBlockEnd = val

    #-----------------------------------------------------------------
    def gTranBlockStart(self):
        return self.__TranBlockStart

    #-----------------------------------------------------------------
    def rTranBlockStart(self, val):
        self.__TranBlockStart = val

    #-----------------------------------------------------------------
    def gTranBlockEnd(self):
        return self.__TranBlockEnd

    #-----------------------------------------------------------------
    def rTranBlockEnd(self, val):
        self.__TranBlockEnd = val

    #-----------------------------------------------------------------
    ## ConvLayer must override this method for real weights
    def gNumberWeights(self):
        return 0

    #-----------------------------------------------------------------
    def gBaseLayerStr(self):
        i = 0
        s = ""
        for prevLayer in self.gPrevLayers():
            ofmap_desc = prevLayer.gOfmapDesc()
            if i == 0:
                s = str(ofmap_desc)
            else:
                s += "+" + str(ofmap_desc)
        s += "-->" + str(self.gOfmapDesc())
        return s

    #-----------------------------------------------------------------
    def gStateSizesStr(self):
        if self.qStoreInSB() :
            nIn= self.gInputStateMemWithoutBatching()
            nOut = self.gOutputStateMemWithoutBatching()
            iState = kstr(nIn)
            oState = kstr(nOut)
            tState = kstr(nIn + nOut)
        else:
            nIn = self.gInputSize()
            iState = "(" + kstr(nIn) + ")"
            nOut = self.gOutputSize()
            oState = "(" + kstr(nOut) + ")"
            tState = "(" + kstr(nIn+nOut) + ")"

        numWeights = self.gNumberWeights()
        nextNumWeights = self.gMaxNextLayerNumberWeights()
        totMem = nIn + nOut + numWeights + nextNumWeights
        return ("  IState=" + (iState)
             + ",  OState=" + (oState)
             + ",  TState=" + (tState)
             + ",  NumWeights=" + kstr(numWeights)
             + ",  TMem=" + kstr(totMem)
                 )


    #-----------------------------------------------------------------
    def rNumberStr(self, numStr):
        self.__NumberStr = numStr

    #-----------------------------------------------------------------
    def gNumberStr(self):
        return self.__NumberStr

    #-----------------------------------------------------------------
    def gNetwork(self):
        return self.__Network

    #-----------------------------------------------------------------
    def gOfmapDesc(self):
        return self.__Ofmap_desc

    #-----------------------------------------------------------------
    def gOfmapSize(self):
        return self.__Ofmap_desc.gMapSize()

    #-----------------------------------------------------------------
    def gNumOfmaps(self):
        return self.__Ofmap_desc.gNumMaps()

    #-----------------------------------------------------------------
    def qDataLayer(self):
        return False



    #-----------------------------------------------------------------
    def gNameType(self):
        return self.gName() + "{" + self.gTypeStr() + "}"

    #-----------------------------------------------------------------
    def gName(self):
        return self.__LayerName

    #-----------------------------------------------------------------

    def gNameNum(self):
        return self.gName() + "-" + self.m_NumStr

    #-----------------------------------------------------------------
    def gDotId(self):
        numStr = self.m_NumStr.replace(".", "_")
        return self.gName() + "_" + numStr

    #-----------------------------------------------------------------
    def gDotLabel(self):
        return '"' + self.gName() + "-" + self.m_NumStr + '"'

    #-----------------------------------------------------------------
    def gNameWithSched(self):
        layer = self
        return (layer.gName()
              + ' lev=[' + str(layer.gEarlyLevel())
              +        ',' + str(layer.gLateLevel()) + '] '
              + ' sched=' + str(layer.gSchedule())
              )

    #-----------------------------------------------------------------
    def gNameWithSchedMem(self):
        #Str = kstr
        Str = Kstr
        if self.gName() == "res3d{Add}":
            x = 3
        if self.qStoreInSB():
            imem = self.gInputStateMemWithoutBatching()
            rmem = self.gResMemWithoutBatching()
            omem = self.gOutputStateMemWithBatching()
            bmem = self.gBatchMem()
            s = (SCHED_MEM_FORMAT) % (
                self.gName(),
                Str(imem), Str(omem), 
                Str(rmem),
                (Str(bmem) + "[" + str(self.gBatchFactor()) + "]"),
                )
        else:
            imem = self.gInputSize()
            omem = self.gOutputSize()
            s = (SCHED_MEM_FORMAT) % (
                self.gName(),
                Str(imem), "("+Str(omem)+")",
                "", "",
                )
        return s

    #-----------------------------------------------------------------
    def gDotIdLabel(self):
        return self.gDotId() + ' [label=' + self.gDotLabel() + '];'

    #-----------------------------------------------------------------
    def gNextSchedLayer(self):
        return self.__NextSchedLayer

    #-----------------------------------------------------------------
    def rNextSchedLayer(self, nextSchedLayer):
        self.__NextSchedLayer = nextSchedLayer

    #-----------------------------------------------------------------
    def gPrevSchedLayer(self):
        return self.__PrevSchedLayer


    #-----------------------------------------------------------------
    def rPrevSchedLayer(self, prevSchedLayer):
        self.__PrevSchedLayer = prevSchedLayer


    #-----------------------------------------------------------------
    def gRefCount(self):
        return self.__RefCount

    #-----------------------------------------------------------------
    def changeRefCount(self, num):
        self.__RefCount += num



    #-----------------------------------------------------------------
    # Does this layer store data in the SB?
    # That depends on
    # - the sinks of this layer
    # - scheduling of the sinks
    # Therefore, the return value can be determined only after scheduling is finished.
    #-----------------------------------------------------------------
    def qStoreInSB(self):
        mySched = self.gSchedule()
        assert(mySched != None)
        nextSchedLayer = self.gNextSchedLayer()
        if not nextSchedLayer: ## output
            return True
        elif nextSchedLayer.qConvLayer() : ## to get to convolution
            return True
        else:
            for nextLayer in self.gNextLayers():
                if nextLayer.gSchedule() > mySched + 1:
                    return True

        assert(self.gNumNextLayers() <= 1)
        return False

    #-----------------------------------------------------------------
    def gPrevSbLayers(self):
        return iter(self.__PrevSbLayers)

    #-----------------------------------------------------------------
    def addPrevSbLayer(self, prevLayer):
        assert(prevLayer.qStoreInSB())
        self.__PrevSbLayers.append(prevLayer)



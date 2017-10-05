from abc             import ABCMeta, abstractmethod

from utils.consts    import  *
from utils.fmapdesc  import OfmapDesc
from utils.funcs     import kstr
import nets.network

##########################################################
class Layer(object): # abstract class
    __metaclass__ = ABCMeta

    #-----------------------------------------------------------------
    def __init__(self, ntwrk, prev_layers, ofmap_desc):
        assert(ntwrk)
        assert(isinstance(ntwrk, nets.network.Network))
        assert(isinstance(ofmap_desc, OfmapDesc))

        self.__Network    = ntwrk
        self.__Ofmap_desc  = ofmap_desc.copy()
        self.__Id = None
        self.__NextSchedLayer = None
        self.__PrevSchedLayer = None

        self.__DenseBlockStart = -1
        self.__DenseBlockEnd   = -1
        self.__TranBlockStart = -1
        self.__TranBlockEnd   = -1

        self.__NextLayers = []
        self.__PrevLayers = []

        self.__schedule = None  ## number in [0, NUM_LAYERS-1]
        # counts the number layers that need to be executed
        # and need this layer's
        self.__RefCount = 0
        self.__TotMem = 0
                            

        assert( (len(prev_layers) == 0) == (self.gLayerType() == LAYER_TYPE_DATA) )
        self.__PrevLayers.extend(prev_layers)
        for prevLayer in prev_layers:
            prevLayer.addNextLayer(self)

        ntwrk.addLayer(self) ## will assign index

    #-----------------------------------------------------------------
    def gLayerId(self):
        return self.__Id

    #-----------------------------------------------------------------
    def rLayerId(self, id):
        self.__Id = id

    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def gLayerType(self):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def verify(self):
        assert(False)

    #-----------------------------------------------------------------
    def qConvLayer(self):
        return self.gLayerType() == LAYER_TYPE_CONV

    #-----------------------------------------------------------------
    def gRawInputStateSize(self, batch=1):
        sz = 0
        for inLayer in self.gPrevLayers():
            sz += inLayer.gRawOutputStateSize()
        return sz

    #-----------------------------------------------------------------
    def gRawOutputStateSize(self, batch=1):
        return self.gNumOfmaps() * self.gOfmapSize() * self.gOfmapSize()

    #-----------------------------------------------------------------
    @abstractmethod
    def gSingleBatchInputStateSize(self, batch=1):
        assert(False)

    #-----------------------------------------------------------------
    @abstractmethod
    def gSingleBatchOutputStateSize(self, batch=1):
        assert(False)

    #-----------------------------------------------------------------
    ## Are the input and the output values different?
    ## I.e., does the layer compute anything?
    ## This is important to estimate the memory size; if
    ## the input and output are different, than total memory
    ## size for this layer is the sumof InputStateSize and
    ## OutputStateSize. If not, the total memory needed for
    ## this layer will be equal to OutputStateSize (which
    ## equals InputStateSize).
    ## For example, concatenation layer does not really
    ## compute anything, just passes two inputs as one output.
    ## The same is true for data layer.

    #-----------------------------------------------------------------
    @abstractmethod
    def qPassThrough(self):
        assert(False)


    #-----------------------------------------------------------------
    def gSchedule(self):
        return self.__schedule

    def rSchedule(self, sch):
        self.__schedule = sch

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
        assert(0 <= idx and idx < len(self.gNextLayers()))
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
    def gSingleBatchTotalStateSize(self):
        isize = self.gSingleBatchInputStateSize()
        osize = self.gSingleBatchOutputStateSize()
        if self.qPassThrough():
            assert(isize == osize)
            return osize
        else:
            return isize + osize

    #-----------------------------------------------------------------
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
        iState = self.gSingleBatchInputStateSize()
        oState = self.gSingleBatchOutputStateSize()
        tState = self.gSingleBatchTotalStateSize()
        numWeights = self.gNumberWeights()
        nextNumWeights = self.gMaxNextLayerNumberWeights()
        totMem = tState + numWeights + nextNumWeights
        return ("  IState=" + kstr(iState)
             + ",  OState=" + kstr(oState)
             + ",  TState=" + kstr(tState)
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
    @staticmethod
    def qHasLayerType(layerType, layers):
        hasType = False
        for layer in layers:
            if layer.gLayerType() == layerType:
                hasType = True
                break
        return hasType

    #-----------------------------------------------------------------
    def qHasNextLayerType(self, layerType):
        return Layer.qHasLayerType(layerType, self.gNextLayers())

    #-----------------------------------------------------------------
    def qHasPrevLayerType(self, layerType):
        return Layer.qHasLayerType(layerType, self.gPrevLayers())

    #-----------------------------------------------------------------
    def qNextSchedLayerOfType(self, layerType):
        nextSchedLayer = self.gNextSchedLayer()
        return nextSchedLayer  and nextSchedLayer.gLayerType() == layerType

    #-----------------------------------------------------------------
    def qPrevSchedLayerOfType(self, layerType):
        prevSchedLayer = self.gPrevSchedLayer()
        return prevSchedLayer  and prevSchedLayer.gLayerType() == layerType

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

    def gNameWithSched(self):
        layer = self
        return (layer.gNameNum()
              + ' lev=[' + str(layer.gEarlyLevel()) + ',' + str(layer.gLateLevel()) + '] '
              + ' sched=' + str(layer.gSchedule())
              )

    def gNameWithSchedMem(self):
        return self.gNameWithSched() + ' mem=' + kstr(self.__TotMem)

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


    def gTotMem(self):
        return self.__TotMem

    def rTotMem(self, mem):
        self.__TotMem = mem


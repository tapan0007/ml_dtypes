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

        self.__DenseBlockStart = -1
        self.__DenseBlockEnd   = -1
        self.__TranBlockStart = -1
        self.__TranBlockEnd   = -1

        self.__NextLayers = []
        self.__PrevLayers = []

        assert( (len(prev_layers) == 0) == (self.gLayerType() == LAYER_TYPE_DATA) )
        self.__PrevLayers.extend(prev_layers)
        for prevLayer in prev_layers:
            prevLayer.addNextLayer(self)

        ntwrk.addLayer(self) ## will assign index

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
    def gPrevLayers(self):
        return self.__PrevLayers

    #-----------------------------------------------------------------
    def gNumPrevLayers(self):
        return len(self.gPrevLayers())

    #-----------------------------------------------------------------
    def gNextLayers(self):
        return self.__NextLayers

    #-----------------------------------------------------------------
    def gNumNextLayers(self):
        return len(self.gNextLayers())

    #-----------------------------------------------------------------
    def gPrevLayer(self, idx):
        assert(0 <= idx and idx < len(self.gPrevLayers()))
        return self.gPrevLayers()[idx]

    #-----------------------------------------------------------------
    def gNextLayer(self, idx):
        assert(0 <= idx and idx < len(self.gNextLayers()))
        return self.gNextLayers()[idx]

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
    def gDotId(self):
        numStr = self.m_NumStr.replace(".", "_")
        return self.gName() + "_" + numStr

    #-----------------------------------------------------------------
    def gDotLabel(self):
        return '"' + self.gName() + "-" + self.m_NumStr + '"'

    #-----------------------------------------------------------------
    def gDotIdLabel(self):
        return self.gDotId() + ' [label=' + self.gDotLabel() + '];'



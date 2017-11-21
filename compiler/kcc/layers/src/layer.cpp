#include "layer.h"
#include "datatype.h"



namespace kcc {

namespace layers {

//----------------------------------------------------------------
Layer::Layer(const Params& params, const FmapDesc&ofmap_desc, const vector<Layer*>& prev_layers)
    : m_LayerName(params.m_LayerName)
    , m_BatchFactor(params.m_BatchFactor)
    , m_Network(params.m_Network)
{
    std::copy(prev_layers.begin(), prev_layers.end(), m_PrevLayers);

    assert(m_BatchFactor >= 1);
    m_OfmapDesc = ofmap_desc;
    self.m_Id = LayerIdNull;

    m_NextSchedLayer   = null;
    m_PrevSchedLayer   = null;

    m_DenseBlockStart  = -1
    m_DenseBlockEnd    = -1
    m_TranBlockStart   = -1
    m_TranBlockEnd     = -1

    m_Schedule         = -1;

    // counts the number layers that need to be executed and need this layer's
    m_RefCount         = 0
    m_ResMemWithBatching = 0
    m_ResMemWithoutBatching = 0

    m_IfmapAddress = StateBufferAddress_Invalid;
    m_OfmapAddress = StateBufferAddress_Invalid;
    m_WeightAddress = StateBufferAddress_Invalid;


#if 0
        assert( (len(prev_layers) == 0) == self.qDataLayer() )
        self.__PrevLayers.extend(prev_layers)
        for prevLayer in prev_layers:
            prevLayer.addNextLayer(self)

        ntwrk.addLayer(self) ## will assign index
#endif

}

//----------------------------------------------------------------
const utils::DataType& gDataType() const
{
    return m_Network->gDataType();
}




#if 0
    #-----------------------------------------------------------------
    def gJson(self):
        prevLayers = []
        for prevLayer in self.gPrevLayers():
            prevLayers.append(prevLayer.gName())

        batch = 1
        x = {
            Layer.layer_name_key : self.gName(),
            Layer.type_key       : self.gTypeStr(),
            Layer.prev_layers_key   : prevLayers,
        }
        if self.gNetwork().gUseDimList():
            x.update({
                Layer.ofmap_key : [1, self.gNumOfmaps(), self.gOfmapHeight(), self.gOfmapWidth()]
            })
        else:
            x.update({
                Layer.number_ofmaps_key : self.gNumOfmaps(),
                Layer.ofmap_width_key   : self.gOfmapWidth(),
                Layer.ofmap_height_key  : self.gOfmapHeight()
            })
        return x

    #-----------------------------------------------------------------

    def combineJson(self, it):
        x = {}
        for y in it:
            x.update(y)
            #x = { **x, **y }
        return x

    #-----------------------------------------------------------------
    #-----------------------------------------------------------------
    def qSubSampleLayer(self):
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
    @classmethod
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
    def gDataType(self):
        return self.__Network.ggDataType()


    #-----------------------------------------------------------------
    def gOutputSize(self):
        wordSize = self.gDataType().gSizeInBytes()
        oneBatchSize = (wordSize * self.gNumOfmaps() *
                        (self.gOfmapWidth() * self.gOfmapHeight()))
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
    ## ConvLayer must override these two methods with correct values
    def gNumberWeights(self):
        return 0

    def gNumberWeightsPerPartition(self):
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
        nextSchedLayer = self.gNextSchedLayer()
        nextNumWeights = (nextSchedLayer.gNumberWeights() if nextSchedLayer else 0)

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
    def gOfmapWidth(self):
        return self.__Ofmap_desc.gMapWidth()

    #-----------------------------------------------------------------
    def gOfmapHeight(self):
        return self.__Ofmap_desc.gMapHeight()

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
        name = self.gNameType()
        if name == "res3d{Add}":
            x = 3
        ofmapStr = (str(self.gNumOfmaps()) + "*" 
                   + str(self.gOfmapWidth()) + "*" + str(self.gOfmapHeigh()) )
        if self.qStoreInSB():
            inMem = self.gInputStateMemWithoutBatching()
            residueMem = self.gResMemWithoutBatching()
            outMem = self.gOutputStateMemWithBatching()
            batchMem = self.gBatchMem()
            batchDelta = "["
            for nextSbLayer in self.gNextSbLayers():
                d = self.gOutputStateMemWithoutBatching() * (nextSbLayer.gBatchFactor() - self.gBatchFactor())
                ##  d = Str(batchMem - nextSbLayer.gBatchMem())
                if batchDelta == "[":
                    batchDelta += Str(d)
                else:
                    batchDelta += "," + Str(d)
            batchDelta += "]"

            s = (SCHED_MEM_FORMAT) % (
                name,
                ofmapStr,
                Str(inMem), Str(outMem),
                Str(residueMem),
                (Str(batchMem) + "[" + str(self.gBatchFactor()) + "]"),
                batchDelta,
                )
        else:
            inMem = self.gInputSize()
            outMem = self.gOutputSize()
            s = (SCHED_MEM_FORMAT) % (
                name,
                ofmapStr,
                Str(inMem), "("+Str(outMem)+")",
                "",  # residueMem
                "",  # batchMem
                "",  # bdelta
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
        assert(self.__RefCount >= -num)
        self.__RefCount += num



    #-----------------------------------------------------------------
    # Does this layer store data in the SB?
    # That depends on
    # - the sinks of this layer
    # - scheduling of the sinks
    # Therefore, the return value can be determined only after scheduling is finished.
    #-----------------------------------------------------------------
    def qStoreInSB(self):
        return True ## this because Dana requires SB area for each operation,
                    ## cannot feed convolution directly to pooling

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
        ##assert(self.qStoreInSB())
        return iter(self.__PrevSbLayers)

    def gNumPrevSbLayers(self):
        return len(self.__PrevSbLayers)

    #-----------------------------------------------------------------
    def gNextSbLayers(self):
        assert(self.qStoreInSB())
        return iter(self.__NextSbLayers)

    def gNumNextSbLayers(self):
        return len(self.__NextSbLayers)

    #-----------------------------------------------------------------
    def addPrevSbLayer(self, prevLayer):
        assert(prevLayer.qStoreInSB())
        self.__PrevSbLayers.append(prevLayer)
        if self.qStoreInSB():
            prevLayer.__NextSbLayers.append(self)

    def gIfmapAddress(self):
        return self.__IfmapAddress

    def gOfmapAddress(self):
        return self.__OfmapAddress

    def gWeightAddress(self):
        return self.__WeightAddress

    def rIfmapAddress(self, address):
        self.__IfmapAddress = address

    def rOfmapAddress(self, address):
        assert(address != None)
        self.__OfmapAddress = address

    def rWeightAddress(self, address):
        assert(address != None)
        self.__WeightAddress = address

    @classmethod
    def gOfmapDescFromJson(klass, layerDict, nn):
        if nn.gUseDimList():
            of = layerDict[Layer.ofmap_key] ##  : [1, self.gNumOfmaps(), self.gOfmapHeight(), self.gOfmapWidth()]
            return OfmapDesc(of[1], (of[2], of[3]) )
        else:
            nOfmaps = layerDict[Layer.number_ofmaps_key]
            ofmapH = layerDict[Layer.ofmap_height_key]
            return OfmapDesc(nOfmaps, (ofmapW, ofmapH))

    @classmethod
    def gLayerNameFromJson(klass, layerDict):
        layerName = layerDict[Layer.layer_name_key]
        return layerName

    @classmethod
    def gPrevLayersFromJson(klass, layerDict, nn):
        prevLayers = []
        prevLayersNames = layerDict[Layer.prev_layers_key]
        for prevLayerName in prevLayersNames:
            prevLayers.append(nn.gLayerByName(prevLayerName))
        return prevLayers

#endif

#if 0
#-----------------------------------------------------------------
Json Layer::gJson()
{
    for (auto prevLayers : m_PrevLayers) {
    prevLayers = []
    for prevLayer in self.gPrevLayers():
        prevLayers.append(prevLayer.gName())

    batch = 1
    x = {
        Layer.layer_name_key : self.gName(),
        Layer.type_key       : self.gTypeStr(),
        Layer.prev_layers_key   : prevLayers,
    }
    if self.gNetwork().gUseDimList():
        x.update({
            Layer.ofmap_key : [1, self.gNumOfmaps(), self.gOfmapHeight(), self.gOfmapWidth()]
        })
    else:
        x.update({
            Layer.number_ofmaps_key : self.gNumOfmaps(),
            Layer.ofmap_width_key   : self.gOfmapWidth(),
            Layer.ofmap_height_key  : self.gOfmapHeight()
        })
    return x


}}
#endif

const char* const Layer::m_LayerNameKey     = "layer_name";
const char* const Layer::m_TypeKey          = "layer_type";
const char* const Layer::m_OfmapKey         = "ofmaps";
const char* const Layer::m_NumberOfmapsKey  = "number_ofmaps";
const char* const Layer::m_OfmapWidthKey    = "ofmap_width";
const char* const Layer::m_OfmapHeightKey   = "ofmap_height";
const char* const Layer::m_PrevKayersKey    = "previous_layers";

}}


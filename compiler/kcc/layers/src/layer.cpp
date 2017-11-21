#include <sstream>

#include "layer.h"
#include "datatype.h"
#include "network.h"



namespace kcc {

namespace layers {

//----------------------------------------------------------------
Layer::Layer(const Params& params, const FmapDesc&ofmap_desc, const vector<Layer*>& prev_layers)
    : m_LayerName(params.m_LayerName)
    , m_Network(params.m_Network)
    , m_NextSchedLayer(nullptr)
    , m_PrevSchedLayer(nullptr)
    , m_IfmapAddress(StateBufferAddress_Invalid)
    , m_OfmapAddress(StateBufferAddress_Invalid)
    , m_WeightAddress(StateBufferAddress_Invalid)
    , m_ResMemWithBatching(0)
    , m_ResMemWithoutBatching(0)
    , m_BatchMemory(0)
    , m_BatchFactor(params.m_BatchFactor)
    , m_Schedule(-1)
    , m_CurrLevel(-1)
    , m_EarlyLevel(-1)
    , m_LateLevel(-1)
    , m_DenseBlockStart(1)
    , m_DenseBlockEnd(1)
    , m_TranBlockStart(1)
    , m_TranBlockEnd(1)
    , m_RefCount(0)
    , m_Id(LayerId_Null)
    , m_OfmapDesc(ofmap_desc)
{
    std::copy(prev_layers.begin(), prev_layers.end(), m_PrevLayers.end());
    assert(m_BatchFactor >= 1);

    // counts the number layers that need to be executed and need this layer's



    assert( (prev_layers.size() == 0) == qDataLayer() );
    std::copy(prev_layers.begin(), prev_layers.end(), m_PrevLayers.end());
    for (auto prevLayer : prev_layers) {
        prevLayer->addNextLayer(this);
    }

    m_Network->addLayer(this); // will assign index
}

//----------------------------------------------------------------
const utils::DataType&
Layer::gDataType() const
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

    //----------------------------------------------------------------
    @abstractmethod
    def __str__(self):
        assert(False)
#endif


//----------------------------------------------------------------
string Layer::gBaseLayerStr() const
{
    int32 i = 0;
    string s = "";
    for (auto prevLayer : gPrevLayers()) {
        const FmapDesc& ofmap_desc = prevLayer->gOfmapDesc();
        if (i == 0) {
            s = ofmap_desc.gString();
        } else {
            s += "+" + ofmap_desc.gString();
        }
        ++i;
    }
    s += "-->" + gOfmapDesc().gString();
    return s;
}

//----------------------------------------------------------------
string Layer::gStateSizesStr() const
{
    int64 nIn, nOut, iState, oState, tState;
    if (qStoreInSB()) {
        nIn     = gInputStateMemWithoutBatching();
        nOut    = gOutputStateMemWithoutBatching();
        iState  = nIn;
        oState  = nOut;
        tState  = nIn + nOut;
    } else {
        nIn     = gInputSize();
        nOut    = 0;
        iState  = nIn;
        nOut    = gOutputSize();
        oState  = nOut;
        tState  = nIn+nOut;
    }

    const int64 numWeights = gNumberWeights();
    const Layer* const nextSchedLayer = gNextSchedLayer();
    const int64 nextNumWeights = nextSchedLayer ?  nextSchedLayer->gNumberWeights() : 0;

    const int64 totMem = nIn + nOut + numWeights + nextNumWeights;
    std::stringstream ss;
    if (qStoreInSB()) {
        ss << "  Istate=" << iState << ", OState=" << oState << ", TState=" << tState
           << ", NumWeights=" << numWeights << ", TMem=" << totMem;
    } else {
        ss << "  Istate=(" << iState << "), Ostate=(" << oState << "), TState=(" << tState
           << "), NumWeights=" << numWeights << ", TMem=(" << totMem << ")";
    }
    return ss.str();
}



#if 0
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


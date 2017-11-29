#include <sstream>

#include "layer.hpp"
#include "datatype.hpp"
#include "network.hpp"



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



//-----------------------------------------------------------------
// Does this layer store data in the SB?
// That depends on
// - the sinks of this layer
// - scheduling of the sinks
// Therefore, the return value can be determined only after scheduling is finished.
//-----------------------------------------------------------------
bool Layer::qStoreInSB() const
{
    return true; // this because Dana requires SB area for each operation,
                 // cannot feed convolution directly to pooling

    const int32 mySched = gSchedule();
    assert(mySched >= 0);
    const Layer* nextSchedLayer = gNextSchedLayer();
    if (!nextSchedLayer) { // output
        return true;
    } else if (nextSchedLayer->qConvLayer()) { // to get to convolution
        return true;
    } else {
        for (const Layer* nextLayer : gNextLayers()) {
            if (nextLayer->gSchedule() > mySched + 1) {
                return true;
            }
        }
    }

    assert(gNumNextLayers() <= 1);
    return false;
}



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



//----------------------------------------------------------------
string Layer::gNameWithSched() const
{
    std::stringstream ss;
    ss  << gName()
        << "lev=[" << gEarlyLevel() << ", " << gLateLevel() << "] "
        << "sched=" << gSchedule();
    return ss.str();
}

//----------------------------------------------------------------
string Layer::gNameWithSchedMem() const
{
    std::stringstream s;
    //Str = kstr
    //Str = Kstr
    string name = gNameType();

    if (qStoreInSB()) {
        std::stringstream ss;
        ss << gNumOfmaps() << "*" << gOfmapWidth() << "*" << gOfmapHeight();
        const auto inMem = gInputStateMemWithoutBatching();
        const auto residueMem = gResMemWithoutBatching();
        const auto outMem = gOutputStateMemWithBatching();
        const auto batchMem = gBatchMem();
        ss << "[";
        bool first = true;
        for (Layer* nextSbLayer : gNextSbLayers()) {
            const auto d = gOutputStateMemWithoutBatching() * (nextSbLayer->gBatchFactor() - gBatchFactor());
            if (first) {
                first = false;
                ss << d;
            } else {
                ss << "," << d;
            }
        }
        ss << "]";

        s   << name
            << inMem << outMem << residueMem
            << batchMem
            << "["
            << gBatchFactor()
            << "]"
            << ss.str();
    } else {
        std::stringstream ss;
        ss << gNumOfmaps() << "*" << gOfmapWidth() << "*" << gOfmapHeight();

        const auto inMem = gInputSize();
        const auto outMem = gOutputSize();
        s   << name
            << inMem << "(" << outMem << ")"
            << ""
            << ""
            << ""
            << ss.str();
    }
    return s.str();
}


vector<Layer*>
Layer::mkLayerVector2(Layer* layer1, Layer* layer2)
{
    vector<Layer*> vec2;
    vec2.push_back(layer1);
    vec2.push_back(layer2);
    return vec2;
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


}} // namespace


#include <sstream>



#include "datatype.hpp"
#include "layer.hpp"
#include "network.hpp"



namespace kcc {
namespace layers {

//----------------------------------------------------------------
Layer::Layer(const Params& params,
        const FmapDesc&ofmap_desc,
        const string& dataDensorDimSemantics,
        const vector<Layer*>& prev_layers)
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
    , m_RefCount(0)
    , m_Id(LayerId_Null)
    , m_OfmapDesc(ofmap_desc)
    , m_DataTensorDimSemantics(dataDensorDimSemantics)
    , m_RefFileName("")
{
    assert(m_BatchFactor >= 1);
    for (auto prevLayer : prev_layers) {
        m_PrevLayers.push_back(prevLayer);
    }

    // counts the number layers that need to be executed and need this layer's
    for (auto prevLayer : prev_layers) {
        prevLayer->addNextLayer(this);
    }

    m_Network->addLayer(this);
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

    const kcc_int32 mySched = gSchedule();
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
    kcc_int32 i = 0;
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
    kcc_int64 nIn, nOut, iState, oState, tState;
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

    const kcc_int64 numWeights = gNumberWeights();
    const Layer* const nextSchedLayer = gNextSchedLayer();
    const kcc_int64 nextNumWeights = nextSchedLayer ?  nextSchedLayer->gNumberWeights() : 0;

    const kcc_int64 totMem = nIn + nOut + numWeights + nextNumWeights;
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

bool
Layer::verify() const
{
    return true;
}

}} // namespace


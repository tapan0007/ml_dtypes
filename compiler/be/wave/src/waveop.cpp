#include <sstream>



#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"
#include "events/inc/events.hpp"

#include "layers/inc/layer.hpp"
#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

//----------------------------------------------------------------
WaveOp::WaveOp (const WaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : m_Name(params.m_WaveOpName)
    // , m_OfmapDesc(params.m_OfmapDesc)
    , m_Order(params.m_Order)
    , m_LayerName(params.m_LayerName)
    , m_Layer(params.m_Layer)
{
    const bool thisIsBarrier = this->qBarrierWaveOp();
    //assert(params.verify());
    for (auto prevWaveOp : prevWaveOps) {
        auto edge = new WaveEdge(prevWaveOp, this);
        this->m_PrevWaveEdges.push_back(edge);
        prevWaveOp->m_SuccWaveEdges.push_back(edge);
        if (thisIsBarrier) {
            prevWaveOp->setHasOutBarrier();
        }
    }
}

//----------------------------------------------------------------
const std::string&
WaveOp::gLayerName () const
{
    return m_LayerName;
}


kcc_int32
WaveOp::gNumberPrevWaitEdges() const
{
    kcc_int32 numWait = 0;
    for (auto prevWaveEdge : m_PrevWaveEdges) {
        if (prevWaveEdge->qNeedToSync()) {
            ++numWait;
        }
    }
    return numWait;
}

kcc_int32
WaveOp::gNumberSuccWaitEdges() const
{
    kcc_int32 numWait = 0;
    for (auto succWaveEdge : m_SuccWaveEdges) {
        if (succWaveEdge->qNeedToSync()) {
            ++numWait;
        }
    }
    return numWait;
}



bool
WaveOp::verify () const
{
    if (m_Name == "") {
        return false;
    }
    if (m_LayerName == "") {
        return false;
    }
    //// OK to have null Layer.
    if (m_Order < 0) {
        return false;
    }
    return true;
}





bool
WaveOp::Params::verify() const
{
    if (m_WaveOpName == "") {
        return false;
    }
    if (m_LayerName == "") {
        return false;
    }
    return true;
}





}} // namespace


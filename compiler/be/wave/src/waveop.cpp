#include <sstream>



#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/passkey.hpp"
#include "events/inc/events.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

/****************************************************************
 *                                                              *
 ****************************************************************/
WaveOp::WaveOp (const WaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : m_Name(params.m_WaveOpName)
    // , m_OfmapDesc(params.m_OfmapDesc)
    , m_Order(params.m_Order)
    , m_LayerName(params.m_LayerName)
{
    //assert(params.verify());
    for (auto prevWaveOp : prevWaveOps) {
        auto edge = new WaveEdge(prevWaveOp, this);
        this->m_PrevWaveEdges.push_back(edge);
        prevWaveOp->m_SuccWaveEdges.push_back(edge);
    }
}

/****************************************************************
 *                                                              *
 ****************************************************************/
const std::string&
WaveOp::gLayerName () const
{
    return m_LayerName;
}


/****************************************************************
 *                                                              *
 ****************************************************************/
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

/****************************************************************
 *                                                              *
 ****************************************************************/
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


/****************************************************************
 *                                                              *
 ****************************************************************/
void
WaveOp::DisconnectPrevEdge(Passkey<nets::Network>, WaveEdge* prevEdge)
{
    kcc_int32 idxFound = -1;
    const kcc_int32 numPrevEdges = m_PrevWaveEdges.size();

    for (kcc_int32 idx = 0; idx < numPrevEdges; ++idx) {
        if (m_PrevWaveEdges[idx] == prevEdge) {
            idxFound = idx;
            break;
        }
    }
    Assert(0 <= idxFound && idxFound < numPrevEdges, "Prev edge not found");
    if (idxFound < numPrevEdges - 1) {
        m_PrevWaveEdges[idxFound] = m_PrevWaveEdges[numPrevEdges - 1];
    }
    m_PrevWaveEdges.pop_back();
    prevEdge->zToOp(utils::Passkey<WaveOp>());
}

/****************************************************************
 *                                                              *
 ****************************************************************/
void
WaveOp::DisconnectSuccEdge(Passkey<nets::Network>, WaveEdge* succEdge)
{
    kcc_int32 idxFound = -1;
    const kcc_int32 numSuccEdges = m_SuccWaveEdges.size();

    for (kcc_int32 idx = 0; idx < numSuccEdges; ++idx) {
        if (m_SuccWaveEdges[idx] == succEdge) {
            idxFound = idx;
            break;
        }
    }
    Assert(0 <= idxFound && idxFound < numSuccEdges, "Succ edge not found");
    if (idxFound < numSuccEdges - 1) {
        m_SuccWaveEdges[idxFound] = m_SuccWaveEdges[numSuccEdges - 1];
    }
    m_SuccWaveEdges.pop_back();
    succEdge->zFromOp(utils::Passkey<WaveOp>());
}


/****************************************************************
 *                                                              *
 ****************************************************************/
void
WaveOp::addPrevWaveEdge(WaveEdge* prevEdge)
{
    m_PrevWaveEdges.push_back(prevEdge);
    if (! prevEdge->gToOp()) {
        prevEdge->rToOp(utils::Passkey<WaveOp>(), this);
    }
}

/****************************************************************
 *                                                              *
 ****************************************************************/
void
WaveOp::addSuccWaveEdge(WaveEdge* succEdge)
{
    m_SuccWaveEdges.push_back(succEdge);
    if (! succEdge->gFromOp()) {
        succEdge->rFromOp(utils::Passkey<WaveOp>(), this);
    }
}

/****************************************************************
 *                                                              *
 ****************************************************************/
bool
WaveOp::verify () const
{
    if (m_Name == "") {
        return false;
    }
    if (m_LayerName == "") {
        return false;
    }
    if (m_Order < 0) {
        return false;
    }
    return true;
}





/****************************************************************
 *                                                              *
 ****************************************************************/
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


/****************************************************************
 *                                                              *
 ****************************************************************/
WaveOp* 
WaveOp::PrevWaveOps::iterator::operator* () const
{
    WaveEdge* prevEdge = m_PrevEdges[m_Idx];
    return prevEdge->gFromOp();
}

/****************************************************************
 *                                                              *
 ****************************************************************/
bool
WaveOp::PrevWaveOps::iterator::operator!= (const iterator& rhs) const
{
    return rhs.m_Idx != m_Idx;
}


/****************************************************************
 *                                                              *
 ****************************************************************/
auto
WaveOp::gPrevWaveops() const -> PrevWaveOps
{
    return PrevWaveOps(this);
}


/****************************************************************
 *                                                              *
 ****************************************************************/
WaveOp* 
WaveOp::SuccWaveOps::iterator::operator* () const
{
    WaveEdge* succEdge = m_SuccEdges[m_Idx];
    return succEdge->gToOp();
}

/****************************************************************
 *                                                              *
 ****************************************************************/
bool
WaveOp::SuccWaveOps::iterator::operator!= (const iterator& rhs) const
{
    return rhs.m_Idx != m_Idx;
}


/****************************************************************
 *                                                              *
 ****************************************************************/
auto
WaveOp::gSuccWaveops() const -> SuccWaveOps
{
    return SuccWaveOps(this);
}


}} // namespace


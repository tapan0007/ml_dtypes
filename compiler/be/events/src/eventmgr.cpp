#include <limits>
#include <sstream>

#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

#include "arch/inc/arch.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wave/inc/barrierwaveop.hpp"

#include "nets/inc/network.hpp"

#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"

kcc::kcc_int32 waveopIdxBrk = 9212;

namespace kcc {
namespace events {


class LoopCmp {
public:
    virtual bool qKeepGoing(kcc_int32 idx) const = 0;
    virtual kcc_int32 change(kcc_int32 idx) const = 0;
};

class LoopCmpBack : public LoopCmp {
public:
    bool qKeepGoing(kcc_int32 idx) const {
        return idx >= 0;
    }
    kcc_int32 change(kcc_int32 idx) const {
        return idx - 1;
    }
};

class LoopCmpForw : public LoopCmp {
public:
    LoopCmpForw(kcc_int32 N)
        : m_N(N)
    {}
public:
    bool qKeepGoing(kcc_int32 idx) const {
        return idx < m_N;
    }
    kcc_int32 change(kcc_int32 idx) const {
        return idx + 1;
    }
private:
    const kcc_int32 m_N;
};





EventMgr::EventMgr(nets::Network& network)
    : m_Network(network)
    , m_EventId(0)
{
    initEventSets();
}


void
EventMgr::initEventSets()
{
    m_Available.clear();
    m_InFlight.clear();
    m_Completed.clear();

    for (EventId eventId = BarrierEvent_FirstNonBarrierEvent; eventId < EventId_Invalid(); ++eventId) {
        m_Available.insert(eventId);
    }
}


EventId
EventMgr::getLocalEventId(const wave::WaveEdge* edge)
{
    const wave::WaveOp* const fromOp = edge->gFromOp();
    const wave::WaveOp* const toOp = edge->gToOp();
    Assert(fromOp != toOp, "From (", fromOp->gName(), ") and to (", toOp->gName(), ") waveops should be different");
    const EventId eventId = m_EventId++;
    if (m_EventId >= EventId_Invalid()) {
        m_EventId = 0;
    }
    return eventId;
}

void
EventMgr::moveCompletedEventsToAvailable()
{
    // Avaliable += Completed;
    for (auto evtId : m_Completed) {
        const auto ret = m_Available.insert(evtId); // ret.second is false if element already exists
        Assert(ret.second, "Event id ", evtId, " already in completed and available event sets");
    }
    m_Completed.clear();
}


// For each succ edge take one available event and assign it
void
EventMgr::assignEventsToNewSuccEdges(wave::WaveOp* waveop)
{
    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToWaitFor()) {
            continue;
        }
        // Available --> InFlight
        Assert(!m_Available.empty(), "Trying to get event from empty available set of events");
        const auto evtId = (*m_Available.begin());

        mvFromAvailableToInFlight(evtId);
        succWaveEdge->rEvent(EventSetMode::OnEndWrDst, evtId, EventWaitMode::SetThenClear);
    }
}

// For each prev edge move evt id from in-flight to completed.
void
EventMgr::completeEventsOnPrevEdges(wave::WaveOp* waveop)
{
    // For each prev edge move evt id from in-flight to completed.
    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToWaitFor()) {
            continue;
        }
        const EventId evtId = prevWaveEdge->gEventId();
        if (prevWaveEdge->qEdgeIsFromBarrier()) {
            Assert(qBarrierEvent(evtId), "Found non-barrier event id ", evtId,
                " on edge from barrier to waveop ", waveop->gName());
            continue;
        }
        mvFromInFlightToCompleted(evtId);
    }
}



void
EventMgr::findWaveopsOnOtherEngines(kcc_int32 waveopIdx,
        EngineId barrierEngId, bool backward,
        std::vector<wave::WaveOp*>& waveops)
{
    std::array<EngineId, NumberRealEngines> engineIds { {
        EngineId::PeArray, EngineId::Activation,
        EngineId::Pooling, EngineId::StreamProc,
        EngineId::DmaEng
    } };

    LoopCmpBack cmpBack;
    LoopCmpForw cmpForw(m_Network.gNumberWaveops());
    LoopCmp& cmp(backward ? static_cast<LoopCmp&>(cmpBack) : static_cast<LoopCmp&>(cmpForw));

    // This could be improved to NOT loop over engines
    for (kcc_int32 k = 0; k < NumberRealEngines; ++k) {
        const auto engId = engineIds[k];
        if (barrierEngId == engId) {
            continue;
        }

        for (kcc_int32 otherIdx = waveopIdx; cmp.qKeepGoing(otherIdx);
             otherIdx = cmp.change(otherIdx))
        {
            const auto otherWaveop = m_Network.gWaveOp(otherIdx);
            if (otherWaveop->qBarrierWaveOp()) {
                break; // do not go beyound another barrier waveop
            }
            const EngineId otherEngId = otherWaveop->gEngineId();
            if (otherEngId == engId) {
                waveops.push_back(otherWaveop);
                break;
            }
        }
    }
}


EngineId
EventMgr::gBarrierEngineId(const wave::WaveOp* /*prevWaveop*/, const wave::WaveOp* /*succWaveop*/)
{
    return EngineId::DmaEng;
}



/***************************************************************
Predecessor of Loading Weights can only be:
Another MatMul which
***************************************************************/
void
EventMgr::processMatMult(wave::MatMulWaveOp* matmulWaveop)
{
    int numPrevs = 0;

    for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToWaitFor()) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveEdge->qNeedToWaitFor()) {
            Assert(prevWaveEdge->gEventId() != EventId_Invalid(),
                    "Need to wait on edge from ", prevWaveEdge->gFromOp()->gName(), " to ", matmulWaveop->gName(),
                    ", but event id is invalid");
        }
        //const EventId eventId = getLocalEventId(prevWaveEdge);
        //prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);

    }
}



/***************************************************************
***************************************************************/
void
EventMgr::processWaveop(wave::WaveOp* waveop)
{
    int numPrevs = 0;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        // when two waveops execute on the same engine, no need for sync
        // except for DMA. If Load depends on a Save, Load must wait on Save
        if (! prevWaveEdge->qNeedToWaitFor()) {
            continue;
        }
        auto prevWaveop = prevWaveEdge->gFromOp();
        ++numPrevs;

        if (prevWaveop->gType() == waveop->gType()) {
            Assert(false, "Predecessors of ", waveop->gTypeStr(),
                " cannot be another waveop of the same type: ", prevWaveop->gName());
        } else {
            if (prevWaveEdge->qNeedToWaitFor()) {
                Assert(prevWaveEdge->gEventId() != EventId_Invalid(),
                        "Need to wait on edge from ", prevWaveop->gName(), " to ", waveop->gName(),
                        ", but event id is invalid");
            }
            //const EventId eventId = getLocalEventId(prevWaveEdge);
            //prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }
    }
}


/***************************************************************
 * Events change state from Available -> InFlight -> Completed -> Available
 * 1. Available -> InFlight  This change occurs when the beginning of an
 *    edge is encountered and the event is assigned to the edge.
 * 2. InFlight -> Completed  This change occurs when the wavegraph sequence 
 *    encounteres the end of an edge with the event.
 * 3. Completed -> Available This change occurs when a barrier is set
 *    All events on the edges that are completed have been consumed
***************************************************************/
void
EventMgr::insertBarriers() {
    //const kcc_int32 NumNonBarrierEvents = EventId_Invalid() - BarrierEvent_FirstNonBarrierEvent; 
    const kcc_int32 numWaveops = m_Network.gNumberWaveops();
    // Need barrier between Waveop[ barrierIndices[j] ] and Waveop[ 1+barrierIndices[j] ]
    //std::vector<kcc_int32> barrierIndices;
    std::vector<wave::WaveOp*> waveopsWithBarriers;

    initEventSets();
    //const auto availEnd(m_Available.end());
    //const auto inflightEnd(m_InFlight.end());
    //const auto completedEnd(m_Completed.end());

    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        if (waveopIdxBrk == waveopIdx) {
            utils::breakFunc(waveopIdx);
        }
        auto waveop = m_Network.gWaveOp(waveopIdx);
        auto immedPrevWaveop = m_Network.gWaveOp(waveopIdx-1);
        kcc_uint64 numSuccEvents = waveop->gNumberSuccWaitEdges();
        if (numSuccEvents > m_Available.size()) {
            // if waveopIdx is included too many events in session,
            // so need barrier between waveop[waveopIdx-1] and waveop[waveopIdx]
            const EngineId barrierEngId = gBarrierEngineId(immedPrevWaveop, waveop);

            std::vector<wave::WaveOp*> prevWaveops;
            findWaveopsOnOtherEngines(waveopIdx-1, barrierEngId, true, prevWaveops);
            Assert(prevWaveops.size() < NumberRealEngines, "Collected too many prev waveops for barrier between waveop ",
                immedPrevWaveop->gName(), " and waveop ", waveop->gName());
            std::vector<wave::WaveOp*> succWaveops;
            findWaveopsOnOtherEngines(waveopIdx, barrierEngId, false, succWaveops);
            Assert(succWaveops.size() < NumberRealEngines, "Collected too many prev waveops for barrier between waveop ",
                immedPrevWaveop->gName(), " and waveop ", waveop->gName());

            std::stringstream barrierWaveopName;
            barrierWaveopName << "barrier_" << waveopIdx-1 << "_" << waveopIdx;
            wave::WaveOp::Params params;
            params.m_WaveOpName    = barrierWaveopName.str();


            const auto barrierWaveop = new wave::BarrierWaveOp(params, prevWaveops, succWaveops, barrierEngId);

            waveopsWithBarriers.push_back(barrierWaveop);
            moveCompletedEventsToAvailable();
            Assert(numSuccEvents <= m_Available.size(), "Not enough event IDs after barrrier");
            assignEventsToBarrier(barrierWaveop);
        }
        assignEventsToNewSuccEdges(waveop);
        completeEventsOnPrevEdges(waveop);
        waveopsWithBarriers.push_back(waveop);
    }

    m_Network.replaceWaveops(waveopsWithBarriers);
}


void
EventMgr::assignEventsToBarrier(wave::BarrierWaveOp* barrierWaveop)
{
    for (auto prevWaveEdge : barrierWaveop->gPrevWaveEdges()) {
        const wave::WaveOp* fromWaveop = prevWaveEdge->gFromOp();
        Assert(!fromWaveop->qBarrierWaveOp(), "Barrier waveop ", barrierWaveop->gName(),
            " has a predecessor that is also a barrier waveop ", fromWaveop->gName());
        Assert(fromWaveop->gEngineId() != barrierWaveop->gEngineId(),
                "Barrier waveop ", barrierWaveop->gName(), " on engine ", 
                static_cast<int>(barrierWaveop->gEngineId()),
                " has a predecessor on the same engine ", fromWaveop->gName());
        const EventId evtId = gEventIdToBarrier(fromWaveop->gEngineId());
        prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, evtId, EventWaitMode::SetThenClear);
    }

    for (auto succWaveEdge : barrierWaveop->gSuccWaveEdges()) {
        const wave::WaveOp* toWaveop = succWaveEdge->gToOp();
        Assert(!toWaveop->qBarrierWaveOp(), "Barrier waveop ", barrierWaveop->gName(),
            " has a successor that is also a barrier waveop ", toWaveop->gName());
        Assert(toWaveop->gEngineId() != barrierWaveop->gEngineId(),
                "Barrier waveop ", barrierWaveop->gName(), " on engine ", 
                static_cast<int>(barrierWaveop->gEngineId()),
                " has a successor on the same engine ", toWaveop->gName());
        const EventId evtId = gEventIdFromBarrier(toWaveop->gEngineId());
        succWaveEdge->rEvent(EventSetMode::OnEndWrDst, evtId, EventWaitMode::SetThenClear);
    }
}


EventId
EventMgr::gEventIdToBarrier(EngineId fromEngId)
{
    switch (fromEngId) {
    case EngineId::PeArray:
        return BarrierEvent_FromPe;
        break;
    case EngineId::Pooling:
        return BarrierEvent_FromAct;
        break;
    case EngineId::Activation:
        return BarrierEvent_FromPool;
        break;
    case EngineId::StreamProc:
        return BarrierEvent_FromStreamProc;
        break;
    case EngineId::DmaEng:
        return BarrierEvent_FromDma;
        break;
    default:
        Assert(false, "Bad from engine id ", static_cast<int>(fromEngId));
        break;
    }
    return BarrierEvent_FromDma;
}


EventId
EventMgr::gEventIdFromBarrier(EngineId toEngId)
{
    switch (toEngId) {
    case EngineId::PeArray:
        return BarrierEvent_ToPe;
        break;
    case EngineId::Pooling:
        return BarrierEvent_ToAct;
        break;
    case EngineId::Activation:
        return BarrierEvent_ToPool;
        break;
    case EngineId::StreamProc:
        return BarrierEvent_ToStreamProc;
        break;
    case EngineId::DmaEng:
        return BarrierEvent_ToDma;
        break;
    default:
        Assert(false, "Bad to engine id ", static_cast<int>(toEngId));
        break;
    }
    return BarrierEvent_ToDma;
}


void
EventMgr::processWaveops()
{
    insertBarriers();

    for (auto waveOp : m_Network.gWaveOps()) {
        if (auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            processMatMult(matmulWaveop);
            continue;
        } else {
            processWaveop(waveOp);
        }
    }
}


/***************************************************************
***************************************************************/
void
EventMgr::mvEventFromSetToSet(EventId evtId, EventSet& fromSet, EventSet& toSet,
        const char* fromStr, const char* toStr)
{
    Assert(qEventRegular(evtId), "Cannot move non-regular event id from ", fromStr, " to ", toStr);
    Assert(fromSet.find(evtId) != fromSet.end(), "Event from prev edge not in ", fromStr);
    Assert(toSet.find(evtId) == toSet.end(), "Event from prev edge already in the ", toStr, " set");
    fromSet.erase(evtId);
    toSet.insert(evtId);
}

void
EventMgr::mvFromInFlightToCompleted(EventId evtId)
{
    mvEventFromSetToSet(evtId, m_InFlight, m_Completed, "InFlight", "Completed");
}

void
EventMgr::mvFromAvailableToInFlight(EventId evtId)
{
    mvEventFromSetToSet(evtId, m_Available, m_InFlight, "Available", "InFlight");
}

void
EventMgr::mvFromCompletedToAvailable(EventId evtId)
{
    mvEventFromSetToSet(evtId, m_Completed, m_Available, "Completed", "Available");
}



} // namespace events
}


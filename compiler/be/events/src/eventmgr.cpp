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
#include "wave/inc/nopwaveop.hpp"

#include "nets/inc/network.hpp"

#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"

kcc::kcc_int32 waveopIdxBrk = 9212;

namespace kcc {
namespace events {






EventMgr::EventMgr(nets::Network& network)
    : m_Network(network)
{
    initEventSets();
}


void
EventMgr::initEventSets()
{
    m_Available.clear();
    m_InFlight.clear();
    m_Completed.clear();

    for (EventId eventId = ReservedEvent_FirstNonReserved; eventId < EventId_Invalid(); ++eventId) {
        m_Available.insert(eventId);
    }
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
        succWaveEdge->rEvent(EventSetMode::OnEndWrDst, evtId, EventWaitMode::WaitThenClear);
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
        const wave::WaveOp* const precWaveop = prevWaveEdge->gFromOp();
        Assert(!precWaveop->qNopWaveOp(),
            "Non-nop waveop ", waveop->gName(), " has incomiing nop-waveop ", precWaveop->gName()); 
        mvFromInFlightToCompleted(evtId);
    }
}



EngineId
EventMgr::gBarrierEngineId()
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
        }
    }
}

wave::NopWaveOp*
EventMgr::mkNopWaveop(wave::WaveOp* prevWaveop, EngineId engId,
                      kcc_int32 nopIdx, kcc_int32 waveopIdx)
{
    std::stringstream nopWaveopName;
    nopWaveopName << "nop_" << nopIdx << "_" << waveopIdx-1 << "_" << waveopIdx;
    wave::NopWaveOp::Params params;
    params.m_WaveOpName    = nopWaveopName.str();
    std::vector<wave::WaveOp*> prevWaveops;
    EventId evtId = events::EventId_Invalid();
    if (prevWaveop) {
        prevWaveops.push_back(prevWaveop);
        evtId = gEventIdBetweenEngines(prevWaveop->gEngineId(), engId);

    }
    const auto nopWaveop = new wave::NopWaveOp(params, prevWaveops, engId, evtId);

    return nopWaveop;
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
    const kcc_int32 numWaveops = m_Network.gNumberWaveops();
    std::vector<wave::WaveOp*> newWaveops;

    initEventSets();

    const std::array<EngineId, 4> engineIds { {
        EngineId::PeArray, EngineId::Activation,
        EngineId::Pooling, EngineId::DmaEng
    } };
    const EngineId barrierEngId = gBarrierEngineId();
    kcc_int32 nopIdx = 0;

    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        if (waveopIdxBrk == waveopIdx) {
            utils::breakFunc(waveopIdx);
        }
        const auto waveop = m_Network.gWaveOp(waveopIdx);
        kcc_uint64 numSuccEvents = waveop->gNumberSuccWaitEdges();
        if (numSuccEvents > m_Available.size()) {
            //
            //  PE   -> EvPeAct ->
            //      ACT  -> EvActPool ->
            //          POOL -> EvPoolDma ->
            //              DMA  -> EvDmaPe ->
            //          PE   -> EvPeAct ->
            //      ACT  -> EvActPool ->
            //  POOL
            wave::WaveOp* prevWaveop = nullptr;

            for (size_t k = 0; k < engineIds.size(); ++k) {
                const auto engId = engineIds[k];
                if (barrierEngId == engId) {
                    continue;
                }
                wave::NopWaveOp* const nopWaveop = mkNopWaveop(prevWaveop, engId, nopIdx, waveopIdx);
                newWaveops.push_back(nopWaveop);
                prevWaveop = nopWaveop;
            }

            { // DMA
                wave::NopWaveOp* const nopWaveop = mkNopWaveop(prevWaveop, barrierEngId, nopIdx, waveopIdx);
                newWaveops.push_back(nopWaveop);
                prevWaveop = nopWaveop;
            }

            for (size_t k = 0; k < engineIds.size(); ++k) {
                const auto engId = engineIds[k];
                if (barrierEngId == engId) {
                    continue;
                }
                wave::NopWaveOp* const nopWaveop = mkNopWaveop(prevWaveop, engId, nopIdx, waveopIdx);
                newWaveops.push_back(nopWaveop);
                prevWaveop = nopWaveop;
            }

            moveCompletedEventsToAvailable();
            Assert(numSuccEvents <= m_Available.size(), "Not enough event IDs after barrrier");

        }
        assignEventsToNewSuccEdges(waveop);
        completeEventsOnPrevEdges(waveop);
        newWaveops.push_back(waveop);
    }

    const kcc_int32 numNewWaveops = newWaveops.size();
    for (kcc_int32 waveopIdx = 0; waveopIdx < numNewWaveops; ++waveopIdx) {
        const auto waveop = newWaveops[waveopIdx];
        verifyWaveop(waveop);
    }

    m_Network.replaceWaveops(newWaveops);
}


void
EventMgr::verifyWaveop(const wave::WaveOp* waveop) const
{
    std::set<EventId> eventIds;
    const auto idSetEnd(eventIds.end());
    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        const EventId evtId = prevWaveEdge->gEventId();
        Assert(idSetEnd == eventIds.find(evtId), "Double event ID ", evtId, " on waveop ", waveop->gName());
    }
    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        const EventId evtId = succWaveEdge->gEventId();
        Assert(idSetEnd == eventIds.find(evtId), "Double event ID ", evtId, " on waveop ", waveop->gName());
    }
}

EventId
EventMgr::gEventIdBetweenEngines(EngineId fromId, EngineId toId)
{
    switch (fromId) {
    case EngineId::Activation:
        switch (toId) {
        case EngineId::DmaEng:
            return ReservedEvent_ActDma;
            break;
        case EngineId::PeArray:
            return ReservedEvent_ActPe;
            break;
        case EngineId::Pooling:
            return ReservedEvent_ActPool;
            break;
        case EngineId::StreamProc:
            return ReservedEvent_ActSp;
            break;
        default:
            Assert(false, "Bad to-engine id ", static_cast<int>(toId));
            break;
        }

    case EngineId::DmaEng:
        switch (toId) {
        case EngineId::Activation:
            return ReservedEvent_DmaAct;
            break;
        case EngineId::PeArray:
            return ReservedEvent_DmaPe;
            break;
        case EngineId::Pooling:
            return ReservedEvent_DmaPool;
            break;
        case EngineId::StreamProc:
            return ReservedEvent_DmaSp;
            break;
        default:
            Assert(false, "Bad to-engine id ", static_cast<int>(toId));
            break;
        }

    case EngineId::PeArray:
        switch (toId) {
        case EngineId::Activation:
            return ReservedEvent_PeAct;
            break;
        case EngineId::DmaEng:
            return ReservedEvent_PeDma;
            break;
        case EngineId::Pooling:
            return ReservedEvent_PePool;
            break;
        case EngineId::StreamProc:
            return ReservedEvent_PeSp;
            break;
        default:
            Assert(false, "Bad to-engine id ", static_cast<int>(toId));
            break;
        }

    case EngineId::Pooling:
        switch (toId) {
        case EngineId::Activation:
            return ReservedEvent_PoolAct;
            break;
        case EngineId::DmaEng:
            return ReservedEvent_PoolDma;
            break;
        case EngineId::PeArray:
            return ReservedEvent_PoolPe;
            break;
        case EngineId::StreamProc:
            return ReservedEvent_PoolSp;
            break;
        default:
            Assert(false, "Bad to-engine id ", static_cast<int>(toId));
            break;
        }

    case EngineId::StreamProc:
        switch (toId) {
        case EngineId::PeArray:
            return ReservedEvent_SpPe;
            break;
        case EngineId::Pooling:
            return ReservedEvent_SpPool;
            break;
        case EngineId::Activation:
            return ReservedEvent_SpAct;
            break;
        case EngineId::DmaEng:
            return ReservedEvent_SpDma;
            break;
        default:
            Assert(false, "Bad to-engine id ", static_cast<int>(toId));
            break;
        }
    default:
        Assert(false, "Bad from-engine id ", static_cast<int>(fromId));
        break;
    }
    return ReservedEvent_FirstNonReserved;
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


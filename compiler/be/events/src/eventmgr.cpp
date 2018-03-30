#include <limits>
#include <sstream>

#include "utils/inc/asserter.hpp"

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


namespace kcc {
namespace events {


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
    for (auto evtId : m_Completed) { // Avaliable += Completed;
        const auto ret = m_Available.insert(evtId); // ret.second is false if element already exists
        Assert(!ret.second, "Event id ", evtId, " already in completed and available event sets");
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
        m_Available.erase(evtId);
        succWaveEdge->rEvent(EventSetMode::OnEndWrDst, evtId, EventWaitMode::SetThenClear);
        m_InFlight.insert(evtId);
    }
}

// For each prev edge move evt id from in-flight to completed.
void
EventMgr::completeEventsOnPrevEdges(wave::WaveOp* waveop)
{
    const auto inflightEnd(m_InFlight.end());
    const auto completedEnd(m_Completed.end());

    // For each prev edge move evt id from in-flight to completed.
    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToWaitFor()) {
            continue;
        }
        // InFlight --> Completed
        const EventId evtId = prevWaveEdge->gEventId();
        Assert(m_InFlight.find(evtId) != inflightEnd, "Event from prev edge not in in-flight set");
        m_InFlight.erase(evtId);
        Assert(m_Completed.find(evtId) != completedEnd, "Event from prev edge not in in-flight set");

        m_Completed.insert(evtId);
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
    std::array<bool, NumberRealEngines> found { {
        false, false, false, false, false
    } };


    for (kcc_int32 k = 0; k < NumberRealEngines; ++k) {
        const auto engId = engineIds[k];
        if (barrierEngId == engId || found[k]) {
            continue;
        }
        if (backward) {
            for (kcc_int32 otherIdx = waveopIdx; otherIdx >= 0; --otherIdx) {
                auto otherWaveop = m_Network.gWaveOp(otherIdx);
                if (otherWaveop->gEngineId() == engId) {
                    waveops.push_back(otherWaveop);
                }
            }
        } else {
            const kcc_int32 N = m_Network.gNumberWaveops();
            for (kcc_int32 otherIdx = waveopIdx; otherIdx < N; ++otherIdx) {
                auto otherWaveop = m_Network.gWaveOp(otherIdx);
                if (otherWaveop->gEngineId() == engId) {
                    waveops.push_back(otherWaveop);
                }
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

        const EventId eventId = getLocalEventId(prevWaveEdge);
        prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);

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
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
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
EventMgr::processWaveops()
{
    // From barrier to barrier

    //const kcc_int32 NumNonBarrierEvents = EventId_Invalid() - BarrierEvent_FirstNonBarrierEvent; 
    const kcc_int32 numWaveops = m_Network.gWaveOps().size();
    // Need barrier between Waveop[ barrierIndices[j] ] and Waveop[ 1+barrierIndices[j] ]
    //std::vector<kcc_int32> barrierIndices;
    std::vector<wave::WaveOp*> waveopsWithBarriers;

    initEventSets();
    //const auto availEnd(m_Available.end());
    //const auto inflightEnd(m_InFlight.end());
    //const auto completedEnd(m_Completed.end());

    kcc_int32 waveopIdx = 0; 
    while (waveopIdx < numWaveops) {
        auto waveop = m_Network.gWaveOp(waveopIdx);
        kcc_uint64 numSuccEvents = waveop->gNumberSuccWaitEdges();
        if (numSuccEvents > m_Available.size()) {
            // if waveopIdx is included too many events in session,
            // so need barrier between waveop[waveopIdx-1] and waveop[waveopIdx]
            const EngineId barrierEngId = gBarrierEngineId(m_Network.gWaveOp(waveopIdx-1), waveop);

            std::vector<wave::WaveOp*> prevWaveops;
            findWaveopsOnOtherEngines(waveopIdx-1, barrierEngId, true, prevWaveops);
            std::vector<wave::WaveOp*> succWaveops;
            findWaveopsOnOtherEngines(waveopIdx, barrierEngId, false, succWaveops);

            std::stringstream barrierWaveopName;
            barrierWaveopName << "barrier_" << waveopIdx-1 << "_" << waveopIdx;
            wave::WaveOp::Params params;
            params.m_WaveOpName    = barrierWaveopName.str();

            auto barrierWaveop = new wave::BarrierWaveOp(params, prevWaveops, succWaveops, barrierEngId);

            waveopsWithBarriers.push_back(barrierWaveop);
            moveCompletedEventsToAvailable();
            Assert(numSuccEvents <= m_Available.size(), "Not enough event IDs after barrrier");
        }
        assignEventsToNewSuccEdges(waveop);
        completeEventsOnPrevEdges(waveop);
        waveopsWithBarriers.push_back(waveop);
    }

    m_Network.replaceWaveops(waveopsWithBarriers);




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



} // namespace events


}


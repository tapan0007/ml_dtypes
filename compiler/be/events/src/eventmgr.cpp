#include <limits>
#include <sstream>

#include "utils/inc/asserter.hpp"
#include "utils/inc/debug.hpp"

#include "arch/inc/arch.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/datamovewaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
//#include "wave/inc/sbatomsavewaveop.hpp"
//#include "wave/inc/barrierwaveop.hpp"
#include "wave/inc/tpbcopywaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

#include "nets/inc/network.hpp"

#include "dma/inc/dmaqueue.hpp"
#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"


namespace kcc {
namespace events {






/***********************************************************************
***********************************************************************/
EventMgr::EventMgr(nets::Network& network)
    : m_Network(network)
    , m_EventState(*this)
{
    m_EventState.reset();
}

/***********************************************************************
***********************************************************************/
EventMgr::~EventMgr()
{
    for (auto que : m_Name2Queue) {
        delete (que).second;
    }
}

/***********************************************************************
***********************************************************************/
// For each succ edge take one available event and assign it
void
EventMgr::assignEventsToNewSuccEdges(wave::WaveOp* waveop)
{
    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToSync()) {
            continue;
        }
        if (qUseEvent(succWaveEdge)) {

            Assert(!m_EventState.availableEmpty(),
                   "Trying to get event from empty available set of events");
            const auto evtId = m_EventState.gFirstAvailable();

            m_EventState.mvFromAvailableToInFlight(evtId);
            succWaveEdge->rEvent(EventSetMode::OnEndWrDst, evtId,
                                 EventWaitMode::WaitThenClear);
        } else {
            succWaveEdge->DoSyncWithSemaphore();
        }
    }
}

/***********************************************************************
***********************************************************************/
// For each prev edge move evt id from in-flight to completed.
void
EventMgr::completeEventsOnPrevEdges(wave::WaveOp* waveop)
{
    // For each prev edge move evt id from in-flight to completed.
    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToSync()) {
            continue;
        }
        if (qUseEvent(prevWaveEdge)) {
            const EventId evtId = prevWaveEdge->gEventId();
            const wave::WaveOp* const precWaveop = prevWaveEdge->gFromOp();
            Assert(!precWaveop->qNopWaveOp(),
                "Non-nop waveop ", waveop->gName(), " has incomiing nop-waveop ", precWaveop->gName());
            m_EventState.mvFromInFlightToCompleted(evtId);
        } else {
            Assert(prevWaveEdge->qSyncedWithSemaphore(), "Need to sync with semaphore not set");
        }
    }
}





/***************************************************************
Predecessor of Loading Weights can only be:
Another MatMul which
***************************************************************/
void
EventMgr::processMatMult(wave::MatMulWaveOp* matmulWaveop)
{
    for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToSync()) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (qUseEvent(prevWaveEdge)) {
            Assert(prevWaveEdge->gEventId() != EventId_Invalid(),
                "Need to wait on edge from ", prevWaveEdge->gFromOp()->gName(), " to ",
                matmulWaveop->gName(), ", but event id is invalid");
        } else {
            Assert(prevWaveEdge->qSyncedWithSemaphore(), "Need to sync with semaphore not set");
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
        if (! prevWaveEdge->qNeedToSync()) {
            continue;
        }
        auto prevWaveop = prevWaveEdge->gFromOp();
        ++numPrevs;

        if (prevWaveop->gType() == waveop->gType()) {
            if (! (waveop->qNopWaveOp()) ) {
                std::cerr << "WARNING: two dependent NOP waveops of the same type("
                    << waveop->gTypeStr() << ") " << prevWaveop->gName() << " --> "
                    << waveop->gName() << "\n";
                //Assert(false, "A predecessor of non-NOP waveop ", waveop->gTypeStr(),
                //    " cannot be another waveop of the same type: ", prevWaveop->gName());
            }
        } else {
            if (prevWaveEdge->qNeedToSync()) {
                if (qUseEvent(prevWaveEdge)) {
                    Assert(prevWaveEdge->gEventId() != EventId_Invalid(),
                            "Need to wait on edge from ", prevWaveop->gName(), " to ",
                            waveop->gName(), ", but event id is invalid");
                } else {
                    Assert(prevWaveEdge->qSyncedWithSemaphore(), "Need to sync with semaphore not set");
                }
            }
        }
    }
}

/***********************************************************************
***********************************************************************/
wave::NopWaveOp*
EventMgr::mkNopWaveop(wave::WaveOp* prevWaveop, EngineId engId,
                      kcc_int32 waveopIdx)
{
    std::stringstream nopWaveopName;
    nopWaveopName << "nop_" << m_NopIdx++ << "_" << waveopIdx-1 << "_" << waveopIdx;
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
***************************************************************/
void
EventMgr::insertOneBarrier(kcc_int32 waveopIdx, std::vector<wave::WaveOp*>& newWaveops)
{

    static const std::array<EngineId, 4> engineIds { {
        EngineId::PeArray,
        EngineId::Activation,
        EngineId::Pooling,
        EngineId::AngelEng
    } };
    const kcc_int32 numEngines = m_Kelf ? 3 : 4;

    //
    //  PE   -> EvPeAct ->-+
    //                     ACT  -> EvActPool ->-+
    //                                          POOL -> EvPoolAngel ->-+
    //                                                                 ANGEL
    //                                          POOL <- EvAngelPool <--+
    //                     ACT  <- EvPoolAct <--+
    //  PE   <- EvActPe <--+
    wave::WaveOp* prevWaveop = nullptr;

    //loop1:
    for (auto k = 0; k < numEngines-1; ++k) { // loop1
        const auto engId = engineIds[k];
        wave::NopWaveOp* const nopWaveop = mkNopWaveop(prevWaveop, engId, waveopIdx);
        newWaveops.push_back(nopWaveop);
        prevWaveop = nopWaveop;
    }

    { // last engine (Pooling in Kelf, Dma with Angel)
        wave::NopWaveOp* const nopWaveop = mkNopWaveop(prevWaveop, engineIds[numEngines-1], waveopIdx);
        newWaveops.push_back(nopWaveop);
        prevWaveop = nopWaveop;
    }

    // loop2 must be in reverse order than loop1
    for (auto k = numEngines - 2; k >= 0; --k) {
        const auto engId = engineIds[k];
        wave::NopWaveOp* const nopWaveop = mkNopWaveop(prevWaveop, engId, waveopIdx);
        newWaveops.push_back(nopWaveop);
        prevWaveop = nopWaveop;
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
EventMgr::insertBarriers()
{
    const kcc_int32 numWaveops = m_Network.gNumberWaveops();
    std::vector<wave::WaveOp*> newWaveops;

    m_EventState.reset();
    m_NopIdx = 0;

    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        const auto waveop = m_Network.gWaveOp(waveopIdx);
        kcc_uint64 numSuccEvents = waveop->gNumberSuccWaitEdges();
        if (numSuccEvents > m_EventState.gNumAvailable()) {
            insertOneBarrier(waveopIdx, newWaveops);
            m_EventState.moveCompletedEventsToAvailable();
            Assert(numSuccEvents <= m_EventState.gNumAvailable(),
                   "Not enough event IDs after barrrier. Required: ",
                   numSuccEvents, ", available: ", m_EventState.gNumAvailable(),
                   ". Total number of TPB events is ", arch::Arch::gArch().gNumberAllTpbEvents(),
                   ". Next waveop is ", waveop->gName());

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

    m_Network.replaceWaveops(newWaveops, false);
}


/***********************************************************************
***********************************************************************/
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

/***********************************************************************
***********************************************************************/
EventId
EventMgr::gEventIdBetweenEngines(EngineId fromId, EngineId toId) const
{
    switch (fromId) {
    case EngineId::Activation:
        switch (toId) {
        case EngineId::Pooling:
            return ReservedEvent_ActPool;
            break;
        case EngineId::PeArray:
            return ReservedEvent_ActPe;
            break;
        default:
            Assert(false, "Bad to-engine id for from engine ",
                   static_cast<int>(fromId), ": ", static_cast<int>(toId));
            break;
        }

    case EngineId::AngelEng:
        switch (toId) {
        case EngineId::Pooling:
            return ReservedEvent_AngelPool;
            break;
        default:
            Assert(false, "Bad to-engine id for from engine ",
                   static_cast<int>(fromId), ": ", static_cast<int>(toId));
            break;
        }

    case EngineId::PeArray:
        switch (toId) {
        case EngineId::Activation:
            return ReservedEvent_PeAct;
            break;
        default:
            Assert(false, "Bad to-engine id for from engine ",
                   static_cast<int>(fromId), ": ", static_cast<int>(toId));
            break;
        }

    case EngineId::Pooling:
        switch (toId) {
        case EngineId::Activation:
            return ReservedEvent_PoolAct;
            break;
        case EngineId::AngelEng:
            return ReservedEvent_PoolAngel;
            break;
        default:
            Assert(false, "Bad to-engine id for from engine ",
                   static_cast<int>(fromId), ": ", static_cast<int>(toId));
            break;
        }

    default:
        Assert(false, "Bad from-engine id ", static_cast<int>(fromId));
        break;
    }
    return this->EventId_FirstNonReserved();
}



/***********************************************************************
***********************************************************************/
void
EventMgr::processWaveops(bool useSem)
{
    m_UseSemaphore = useSem;
    if (useSem) {
        determineQueuesAndSemaphoreValues();
    }
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

/***********************************************************************
***********************************************************************/
void
EventMgr::determineQueuesAndSemaphoreValues()
{
    for (auto waveop : m_Network.gWaveOps()) {
        if (! waveop->qDataMoveWaveOp()) {
            continue;
        }
        {
            const auto datamoveWop = dynamic_cast<wave::DataMoveWaveOp*>(waveop);
            Assert(datamoveWop, "Waveop ", waveop->gName(), " expected to be DataMove");


            const dma::DmaQueue* que0(findQueue(datamoveWop, true));
            datamoveWop->rDmaQueue(que0);

            const auto it = m_DmaQueueCount.find(que0);
            kcc_int32 n = -1;
            if (it == m_DmaQueueCount.end()) {
                n = 1;
                m_DmaQueueCount[que0] = n;
            } else {
                ++(*it).second;
                n = (*it).second;
            }
            datamoveWop->rTriggerOrd(n);
        }

        //------------------------------------
        {
            const auto sbatomloadWop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveop);
            if (!sbatomloadWop || !sbatomloadWop->qContainWeights()) {
                continue;
            }
            if (sbatomloadWop->gNumPartitions() <= 1) {
                continue;
            }
            const dma::DmaQueue* que1(findQueue(sbatomloadWop, false));
            sbatomloadWop->rDmaQueue1(que1);

            const auto it = m_DmaQueueCount.find(que1);
            kcc_int32 n = -1;
            if (it == m_DmaQueueCount.end()) {
                n = 1;
                m_DmaQueueCount[que1] = n;
            } else {
                ++(*it).second;
                n = (*it).second;
            }
            sbatomloadWop->rTriggerOrd1(n);
        }
    }
}

/***********************************************************************
***********************************************************************/
const dma::DmaQueue*
EventMgr::findQueue(const wave::DataMoveWaveOp* datamoveWaveop, bool firstQueue)
{
    dma::DmaQueue::QueueType typ = dma::DmaQueue::QueueType::None;
    const EngineId engId = datamoveWaveop->gEngineId();
    Assert(engId != EngineId::None, "Bad engine id for SbAtom ", datamoveWaveop->gName());

    const char* engName = nullptr;
    switch (engId) {
    case EngineId::Pooling:
        engName = "Pool";
        break;
    case EngineId::Activation:
        engName = "Act";
        break;
    case EngineId::PeArray:
        engName = "Pe";
        break;
    case EngineId::AngelEng:
        Assert(false, "Cannot have Angel engine for DMA queue");
        engName = "Dma";
        break;
    default:
        Assert(false, "DMA queue needs Pool, Act, or PeArray engine. It is ",
               static_cast<kcc_int32>(engId));
        engName = nullptr;
        break;
    }
    // Examples:
    // qPoolInW0 - input queue 0 for weights triggered by PoolEng
    // qActOut - output queue triggered by ActEng
    // qPoolS2s - SBUF to SBUF queue triggered by PoolEng
    std::string queName("q");
    queName += engName;
    if (datamoveWaveop->qSbAtomLoadWaveOp()) {
        auto loadWop = dynamic_cast<const wave::SbAtomLoadWaveOp*>(datamoveWaveop);
        if (loadWop->qContainWeights()) {
            typ = dma::DmaQueue::QueueType::Weights;
            if (firstQueue) {
                queName += "InW0";
            } else {
                queName += "InW1";
            }
        } else {
            typ = dma::DmaQueue::QueueType::Input;
            queName += "InD";
        }
    } else if (datamoveWaveop->qSbAtomSaveWaveOp()) {
        queName += "Out";
        typ = dma::DmaQueue::QueueType::Output;
    } else if (datamoveWaveop->qTpbCopyWaveOp()) {
        queName += "S2s";
        typ = dma::DmaQueue::QueueType::SbufToSbuf;
    } else {
        Assert(false, "DmaQueue is used only for Load, Save, or TpbCopy");
    }

    const auto it = m_Name2Queue.find(queName);
    if (it == m_Name2Queue.end()) {
        const auto semId = ReservedSemaphore_FirstNonReserved + m_Name2Queue.size();
        const auto que = new dma::DmaQueue(queName, engId, typ, semId, firstQueue);
        m_Name2Queue[queName] = que;
        return que;
    } else {
        return (*it).second;
    }
}

/***********************************************************************
***********************************************************************/
bool
EventMgr::qUseEvent(const wave::WaveEdge* edge) const
{
    return qUseEventsOnly() || !edge->qCanSyncWithSemaphore();
}

}}


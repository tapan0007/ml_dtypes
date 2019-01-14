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
#include "wave/inc/sbatomsavewaveop.hpp"
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

//***********************************************************************
EventSetMode
EventMgr::gEventSetMode(const wave::WaveOp* waveop, const EventSetMode setMode)
{
    static EventSetMode translateTable[kcc_int32(WaveOpType::Count)][4];  // 4 set modes
    static bool initialized = false;

    if (! initialized) {
        const kcc_int32 onRd   = static_cast<kcc_int32>(EventSetMode::OnEndRdSrc);
        const kcc_int32 onWr   = static_cast<kcc_int32>(EventSetMode::OnEndWrDst);
        const kcc_int32 onDone = static_cast<kcc_int32>(EventSetMode::OnEndInstr);
        const kcc_int32 dontSet = static_cast<kcc_int32>(EventSetMode::DontSet);

        const kcc_int32 typeCnt = static_cast<kcc_int32>(WaveOpType::Count);

        // Start with 1-1 translation
        for (kcc_int32 t = 0; t < typeCnt; ++t) {
            translateTable[t][dontSet]  = EventSetMode::DontSet;
            translateTable[t][onRd]     = EventSetMode::OnEndRdSrc;
            translateTable[t][onWr]     = EventSetMode::OnEndWrDst;
            translateTable[t][onDone]   = EventSetMode::OnEndInstr;
        }
        // Exceptions:
        translateTable[kcc_int32(WaveOpType::Pool)][onDone]             = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::Reciprocal)][onDone]       = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::MatMul)][onDone]           = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::Activation)][onDone]       = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::TensorTensor)][onDone]     = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::TensorScalar)][onDone]     = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::TensorScalarPtr)][onDone]  = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::ScaleAdd)][onDone]         = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::ClipByValue)][onDone]      = EventSetMode::OnEndWrDst;

        translateTable[kcc_int32(WaveOpType::RegLoad)][onDone]          = EventSetMode::OnEndRdSrc;
        translateTable[kcc_int32(WaveOpType::RegLoad)][onWr]            = EventSetMode::OnEndRdSrc;

        translateTable[kcc_int32(WaveOpType::RegStore)][onDone]         = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::RegStore)][onRd]           = EventSetMode::OnEndWrDst;

        translateTable[kcc_int32(WaveOpType::RegShuffle)][onRd]         = EventSetMode::OnEndInstr;
        translateTable[kcc_int32(WaveOpType::RegShuffle)][onWr]         = EventSetMode::OnEndInstr;
        translateTable[kcc_int32(WaveOpType::Nop)][onRd]                = EventSetMode::OnEndInstr;
        translateTable[kcc_int32(WaveOpType::Nop)][onWr]                = EventSetMode::OnEndInstr;

        // These are really semaphores
        translateTable[kcc_int32(WaveOpType::Load)][onDone]             = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::Load)][onRd]               = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::Save)][onDone]             = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::Save)][onRd]               = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::TpbCopy)][onDone]          = EventSetMode::OnEndWrDst;
        translateTable[kcc_int32(WaveOpType::TpbCopy)][onRd]            = EventSetMode::OnEndWrDst;
    }

    
    const kcc_int32 typeIdx = static_cast<kcc_int32>(waveop->gType());
    const kcc_int32 modeIdx = static_cast<kcc_int32>(setMode);
    return translateTable[typeIdx][modeIdx];
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
            succWaveEdge->rEvent(gEventSetMode(waveop, EventSetMode::OnEndWrDst),
                                 evtId,
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
            Assert(! precWaveop->qPartOfBarrier(),
                "Non-Nop waveop ", waveop->gName(), " has incoming barrier Nop-waveop ", precWaveop->gName());
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
                std::cerr << "WARNING: two dependent waveops of the same type("
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
    const auto nopWaveop = new wave::NopWaveOp(params, prevWaveops, engId, evtId, 
                                               wave::NopWaveOp::NopType::Barrier);

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
***************************************************************/
void
EventMgr::linkBarrierNops(std::vector<wave::WaveOp*>& newWaveops)
{
    const kcc_int32 numWaveops = newWaveops.size();
    for (kcc_int32 waveopIdx = 0; waveopIdx < numWaveops; ++waveopIdx) {
        const auto waveop = newWaveops[waveopIdx];
        if (! waveop->qPartOfBarrier()) {
            continue;
        }
        const EngineId engId = waveop->gEngineId();

        // PREDECESSOR
        // is one of pred Waveops on engine already a pred
        wave::WaveOp* predOnSameEng = nullptr;
        for (auto predWop : waveop->gPrevWaveops()) {
            if (predWop->gEngineId() == engId) {
                predOnSameEng = predWop;
                break;
            }
        }
        if (! predOnSameEng) {
            for (kcc_int32 prevIdx = waveopIdx - 1; prevIdx >= 0; --prevIdx) {
                const auto prevWaveop = newWaveops[prevIdx];
                if (prevWaveop->gEngineId() != engId) {
                    continue;
                }
                // found 
                auto edge = new wave::WaveEdge(prevWaveop, waveop);
                prevWaveop->addSuccWaveEdge(edge);
                waveop->addPrevWaveEdge(edge);
                break;
            }
        }


        //SUCCESSOR
        wave::WaveOp* succOnSameEng = nullptr;
        for (auto succWop : waveop->gSuccWaveops()) {
            if (succWop->gEngineId() == engId) {
                succOnSameEng = succWop;
                break;
            }
        }
        if (! succOnSameEng) {
            for (kcc_int32 succIdx = waveopIdx + 1; succIdx < numWaveops; ++succIdx) {
                const auto succWaveop = newWaveops[succIdx];
                if (succWaveop->gEngineId() != engId) {
                    continue;
                }
                // found 
                auto edge = new wave::WaveEdge(waveop, succWaveop);
                waveop->addSuccWaveEdge(edge);
                succWaveop->addPrevWaveEdge(edge);
                break;
            }
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

    linkBarrierNops(newWaveops);
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
    m_Network.RewireMultiOutEdgesOfMatMults();

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
                queName += "W0";
            } else {
                queName += "W1";
            }
        } else {
            if (! loadWop->qTmpBuffer()) {
                typ = dma::DmaQueue::QueueType::Input;
                queName += "In";
            } else {
                typ = dma::DmaQueue::QueueType::TmpToSbuf;
                queName += "R2S";
            }
        }
    } else if (datamoveWaveop->qSbAtomSaveWaveOp()) {
        auto saveWop = dynamic_cast<const wave::SbAtomSaveWaveOp*>(datamoveWaveop);
        if (saveWop->qTmpBuffer()) {
            queName += "Out";
            typ = dma::DmaQueue::QueueType::Output;
        } else {
            queName += "S2R";
            typ = dma::DmaQueue::QueueType::SbufToTmp;
        }
    } else if (datamoveWaveop->qTpbCopyWaveOp()) {
        queName += "S2S";
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


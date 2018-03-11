#include <limits>

#include "utils/inc/asserter.hpp"
#include "utils/inc/events.hpp"

#include "wave/inc/waveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomfilewaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"

#include "nets/inc/network.hpp"

#include "events/inc/eventmgr.hpp"


namespace kcc {
namespace events {


EventMgr::EventMgr(const nets::Network& network)
    : m_Network(network)
    , m_EventId(0)
{
}

kcc_int32
EventMgr::getLocalEventId(const wave::WaveOp* from, const wave::WaveOp* to)
{
    Assert(from != to, "From (", from->gName(), ") and to (", to->gName(), ") events should be different");
    const kcc_int32 eventId = m_EventId++;
    if (m_EventId >= 256) {
        m_EventId = 0;
    }
    return eventId;
}


void
EventMgr::processSuccSaves(wave::WaveOp* waveop)
{
    const EngineId engineId = waveop->gEngineId();
    int smallestAtomSaveOrder = std::numeric_limits<int>::max();
    wave::SbAtomSaveWaveOp* smallestAtomSaveWaveop = nullptr;
#if 0
    int largestAtomSaveOrder = -1;
    wave::SbAtomSaveWaveOp* largestAtomSaveWaveop = nullptr;
#endif

    for (auto succWaveop : waveop->gSuccWaveOps()) {
        if (succWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        if (auto sbatomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(succWaveop)) {
            if (sbatomSaveWaveop->gOrder()  < smallestAtomSaveOrder) {
                smallestAtomSaveOrder = sbatomSaveWaveop->gOrder();
                smallestAtomSaveWaveop = sbatomSaveWaveop;
            }
#if 0
            if (sbatomSaveWaveop->gOrder()  > largestAtomSaveOrder) {
                largestAtomSaveOrder = sbatomSaveWaveop->gOrder();
                largestAtomSaveWaveop = sbatomSaveWaveop;
            }
#endif

            continue;
        }
    }

    if (smallestAtomSaveWaveop) {
        const EventId eventId = getLocalEventId(smallestAtomSaveWaveop, waveop);
        waveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
        smallestAtomSaveWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
    }
}

/***************************************************************
Predecessor of Loading Weights can only be:
Another MatMul which
***************************************************************/
void
EventMgr::processMatMult(wave::MatMulWaveOp* matmulWaveop)
{
    const EngineId engineId = matmulWaveop->gEngineId();
    int numPrevAtomFile = 0;
    int numPrevPool = 0;
    int numPrevActivation = 0;

    int largestAtomFileIfmapOrder = -1;
    wave::SbAtomFileWaveOp* largestSbAtomFileIfmapWaveop = nullptr;
    int largestAtomFileWeightOrder = -1;
    wave::SbAtomFileWaveOp* largestSbAtomFileWeightWaveop = nullptr;

    for (auto prevWaveop : matmulWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (auto sbatomFileWaveop = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            ++numPrevAtomFile;
            if (sbatomFileWaveop->qContainWeights()) {
                if (sbatomFileWaveop->gOrder()  > largestAtomFileWeightOrder) {
                    largestAtomFileWeightOrder = sbatomFileWaveop->gOrder();
                    largestSbAtomFileWeightWaveop = sbatomFileWaveop;
                }
            } else {
                if (sbatomFileWaveop->gOrder()  > largestAtomFileIfmapOrder) {
                    largestAtomFileIfmapOrder = sbatomFileWaveop->gOrder();
                    largestSbAtomFileIfmapWaveop = sbatomFileWaveop;
                }
            }

            continue;
        }

        if (auto poolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            const EventId eventId = getLocalEventId(poolWaveop, matmulWaveop);
            poolWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            matmulWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }

        if (auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            const EventId eventId = getLocalEventId(activationWaveop, matmulWaveop);
            activationWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            matmulWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of MatMult waveop must be one of DramLoad, Pool, Activation: ",
                prevWaveop->gTypeStr());
    }


    if (largestSbAtomFileWeightWaveop) {
        const EventId eventId = getLocalEventId(largestSbAtomFileWeightWaveop, matmulWaveop);
        largestSbAtomFileWeightWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
        matmulWaveop->rLwWaitEvent(eventId, EventWaitMode::SetThenClear);
    }
    if (largestSbAtomFileIfmapWaveop) {
        const EventId eventId = getLocalEventId(largestSbAtomFileIfmapWaveop, matmulWaveop);
        largestSbAtomFileIfmapWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
        matmulWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
    }

    if (matmulWaveop->qStartTensorCalc()) {
        Assert(numPrevAtomFile + numPrevPool + numPrevActivation >= 1,
            "MatMul waveop starting tensor calc should have at least one predecessor from another engine. Num Prev: AtomFile: ",
            numPrevAtomFile, ", Pool: ", numPrevPool, ", Act: ", numPrevActivation);
    }

    processSuccSaves(matmulWaveop);
}

/***************************************************************
***************************************************************/
void
EventMgr::processPool(wave::PoolWaveOp* poolWaveop)
{
    const EngineId engineId = poolWaveop->gEngineId();
    int numPrevAtomFile = 0;
    int numPrevMatMul = 0;
    int numPrevActivation = 0;

    for (auto prevWaveop : poolWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (auto sbatomFileWaveop = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            ++numPrevAtomFile;
            const EventId eventId = getLocalEventId(sbatomFileWaveop, poolWaveop);
            sbatomFileWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            poolWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
            ++numPrevMatMul;
            const EventId eventId = getLocalEventId(matmulWaveop, poolWaveop);
            matmulWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            poolWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            const EventId eventId = getLocalEventId(activationWaveop, poolWaveop);
            activationWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            poolWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of Pool waveop must be one of DramLoad, MatMul, Activation: ",
                prevWaveop->gTypeStr());
    }

    processSuccSaves(poolWaveop);
}


/***************************************************************
***************************************************************/
void
EventMgr::processActivation(wave::ActivationWaveOp* activationWaveop)
{
    const EngineId engineId = activationWaveop->gEngineId();
    int numPrevAtomFile = 0;
    int numPrevPool = 0;
    int numPrevMatMul = 0;

    for (auto prevWaveop : activationWaveop->gPrevWaveOps()) {
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (auto sbatomFileWaveop = dynamic_cast<wave::SbAtomFileWaveOp*>(prevWaveop)) {
            ++numPrevAtomFile;
            const EventId eventId = getLocalEventId(sbatomFileWaveop, activationWaveop);
            sbatomFileWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            activationWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
            ++numPrevMatMul;
            const EventId eventId = getLocalEventId(matmulWaveop, activationWaveop);
            matmulWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            activationWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (auto poolWaveop = dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            const EventId eventId = getLocalEventId(poolWaveop, activationWaveop);
            poolWaveop->rSetEvent(eventId, EventSetMode::OnEndWrDst);
            activationWaveop->rWaitEvent(eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of Pool waveop must be one of DramLoad, MatMul, Activation: ",
                prevWaveop->gTypeStr());
    }

    processSuccSaves(activationWaveop);
}

void
EventMgr::processSbAtomSave(wave::SbAtomSaveWaveOp* sbatomSaveWaveop)
{
    Assert(sbatomSaveWaveop, "Nil SbAtomSaveWaveOp");
}


void
EventMgr::processResAdd(wave::ResAddWaveOp* resaddWaveop)
{
    Assert(resaddWaveop, "Nil ResAddWaveOp");
}



/***************************************************************
***************************************************************/
void
EventMgr::processWaveops()
{
    for (auto waveOp : m_Network.gWaveOps()) {
        if (auto matmulWaveop = dynamic_cast<wave::MatMulWaveOp*>(waveOp)) {
            processMatMult(matmulWaveop);
            continue;
        }
        if (auto poolWaveop = dynamic_cast<wave::PoolWaveOp*>(waveOp)) {
            processPool(poolWaveop);
            continue;
        }
        if (auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(waveOp)) {
            processActivation(activationWaveop);
            continue;
        }
        if (auto sbatomSaveWaveop = dynamic_cast<wave::SbAtomSaveWaveOp*>(waveOp)) {
            processSbAtomSave(sbatomSaveWaveop);
            continue;
        }
        if (auto sbatomFileWaveop = dynamic_cast<wave::SbAtomFileWaveOp*>(waveOp)) {
            Assert(sbatomFileWaveop->gPrevWaveOps().size() == 0,
                "SbAtomFile should have 0 predecessors. Waveop: ", sbatomFileWaveop->gName(),
                ", number predecessors: ", sbatomFileWaveop->gPrevWaveOps().size());
            continue; // when two waveops execute on the same engine, no need for sync
        }
        if (auto resaddWaveop = dynamic_cast<wave::ResAddWaveOp*>(waveOp)) {
            processResAdd(resaddWaveop);
            continue;
        }
        Assert(false, "WaveOp ", waveOp->gName(), " of type ", waveOp->gTypeStr(),
                " not expected");
    }
}


/***************************************************************
***************************************************************/



} // namespace events

int
eventWaitMode2Int(EventWaitMode mode)
{
    switch(mode) {
    case EventWaitMode::NoEvent:
        return NO_WAIT_EVENT;
        break;
    case EventWaitMode::SetOnly:
        return WAIT_EVENT_SET;
        break;
    case EventWaitMode::SetThenClear:
        return WAIT_EVENT_SET_THEN_CLEAR;
        break;
    }
    Assert(false, "Wrong EventWaitMode: ", static_cast<kcc_int32>(mode));
    return 0;
}

int
eventSetMode2Int(EventSetMode mode)
{
    switch(mode) {
    case EventSetMode::NoEvent:
        return NO_SET_EVENT;
        break;
    case EventSetMode::OnEndRdSrc:
        return SET_EVENT_ON_END_RD_SRC;
        break;
    case EventSetMode::OnEndWrDst:
        return SET_EVENT_ON_END_WR_DST;
        break;
    }
    Assert(false, "Wrong EventSetMode: ", static_cast<kcc_int32>(mode));
    return 0;
}


}


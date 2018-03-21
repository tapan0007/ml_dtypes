#include <limits>

#include "utils/inc/asserter.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/resaddwaveop.hpp"

#include "nets/inc/network.hpp"

#include "events/inc/events.hpp"
#include "events/inc/eventmgr.hpp"


namespace kcc {
namespace events {


EventMgr::EventMgr(const nets::Network& network)
    : m_Network(network)
    , m_EventId(0)
{
}

EventId
EventMgr::getLocalEventId(const wave::WaveEdge* edge)
{
    const wave::WaveOp* const fromOp = edge->gFromOp();
    const wave::WaveOp* const toOp = edge->gToOp();
    Assert(fromOp != toOp, "From (", fromOp->gName(), ") and to (", toOp->gName(), ") waveops should be different");
    const kcc_int32 eventId = m_EventId++;
    if (m_EventId >= 256) {
        m_EventId = 0;
    }
    return eventId;
}




/***************************************************************
Predecessor of Loading Weights can only be:
Another MatMul which
***************************************************************/
void
EventMgr::processMatMult(wave::MatMulWaveOp* matmulWaveop)
{
    const EngineId engineId = matmulWaveop->gEngineId();
    int numPrevAtomLoad = 0;
    int numPrevPool = 0;
    int numPrevActivation = 0;

    int largestAtomLoadIfmapOrder = -1;
    wave::WaveEdge* largestSbAtomLoadIfmapWaveEdge = nullptr;
    int largestAtomLoadWeightOrder = -1;
    wave::WaveEdge* largestSbAtomLoadWeightWaveEdge = nullptr;

    for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (auto sbatomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
            ++numPrevAtomLoad;
            if (sbatomLoadWaveop->qContainWeights()) {
                if (sbatomLoadWaveop->gOrder()  > largestAtomLoadWeightOrder) {
                    largestAtomLoadWeightOrder = sbatomLoadWaveop->gOrder();
                    largestSbAtomLoadWeightWaveEdge = prevWaveEdge;
                }
            } else {
                if (sbatomLoadWaveop->gOrder()  > largestAtomLoadIfmapOrder) {
                    largestAtomLoadIfmapOrder = sbatomLoadWaveop->gOrder();
                    largestSbAtomLoadIfmapWaveEdge = prevWaveEdge;
                }
            }

            continue;
        }

        if (dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }

        if (dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of MatMult waveop must be one of DramLoad, Pool, Activation: ",
                prevWaveop->gTypeStr());
    }


    if (largestSbAtomLoadWeightWaveEdge) {
        const EventId eventId = getLocalEventId(largestSbAtomLoadWeightWaveEdge);
        largestSbAtomLoadWeightWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
    }
    if (largestSbAtomLoadIfmapWaveEdge) {
        const EventId eventId = getLocalEventId(largestSbAtomLoadIfmapWaveEdge);
        largestSbAtomLoadIfmapWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
    }

    if (matmulWaveop->qStartTensorCalc()) {
        Assert(numPrevAtomLoad + numPrevPool + numPrevActivation >= 1,
            "MatMul waveop starting tensor calc should have at least one predecessor from another engine. Num Prev: AtomLoad: ",
            numPrevAtomLoad, ", Pool: ", numPrevPool, ", Act: ", numPrevActivation);
    }
}

/***************************************************************
***************************************************************/
void
EventMgr::processPool(wave::PoolWaveOp* poolWaveop)
{
    const EngineId engineId = poolWaveop->gEngineId();
    int numPrevAtomLoad = 0;
    int numPrevMatMul = 0;
    int numPrevActivation = 0;

    for (auto prevWaveEdge : poolWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
            ++numPrevAtomLoad;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
            ++numPrevMatMul;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of Pool waveop must be one of DramLoad, MatMul, Activation: ",
                prevWaveop->gTypeStr());
    }
}


/***************************************************************
***************************************************************/
void
EventMgr::processActivation(wave::ActivationWaveOp* activationWaveop)
{
    const EngineId engineId = activationWaveop->gEngineId();
    int numPrevAtomLoad = 0;
    int numPrevPool = 0;
    int numPrevMatMul = 0;

    for (auto prevWaveEdge : activationWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }

        if (dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
            ++numPrevAtomLoad;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
            ++numPrevMatMul;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of Activation waveop must be one of DramLoad, MatMul, Pool: ",
                prevWaveop->gTypeStr());
    }
}


void
EventMgr::
processSbAtomLoad(wave::SbAtomLoadWaveOp* sbatomLoadWaveop)
{
    const EngineId engineId = sbatomLoadWaveop->gEngineId();
    int numPrevPool = 0;
    int numPrevMatMul = 0;
    int numPrevActivation = 0;
    int numPrevSaves = 0;

    for (auto prevWaveEdge : sbatomLoadWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        if (dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
            ++numPrevMatMul;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::SbAtomSaveWaveOp*>(prevWaveop)) {
            ++numPrevSaves;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of SbAtomLoad waveop must be one of Pool, MatMul, Activation, SbAtomSave: ",
                prevWaveop->gTypeStr());
    }
}


void
EventMgr::processSbAtomSave(wave::SbAtomSaveWaveOp* sbatomSaveWaveop)
{
    const EngineId engineId = sbatomSaveWaveop->gEngineId();
    int numPrevPool = 0;
    int numPrevMatMul = 0;
    int numPrevActivation = 0;
    int numPrevLoads = 0;

    for (auto prevWaveEdge : sbatomSaveWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        if (dynamic_cast<wave::MatMulWaveOp*>(prevWaveop)) {
            ++numPrevMatMul;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::PoolWaveOp*>(prevWaveop)) {
            ++numPrevPool;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::ActivationWaveOp*>(prevWaveop)) {
            ++numPrevActivation;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        if (dynamic_cast<wave::SbAtomLoadWaveOp*>(prevWaveop)) {
            ++numPrevLoads;
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
            continue;
        }
        Assert(false,
                "Predecessors of SbAtomSave waveop must be one of Pool, MatMul, Activation, SbAtomLoad: ",
                prevWaveop->gTypeStr());
    }
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
        if (auto sbatomLoadWaveop = dynamic_cast<wave::SbAtomLoadWaveOp*>(waveOp)) {
            processSbAtomLoad(sbatomLoadWaveop);
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


}


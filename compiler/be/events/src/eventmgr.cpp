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
    int numPrevs = 0;

    for (auto prevWaveEdge : matmulWaveop->gPrevWaveEdges()) {
        const auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        const EventId eventId = getLocalEventId(prevWaveEdge);
        prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);

    }
}

#if 0
/***************************************************************
***************************************************************/
void
EventMgr::processPool(wave::PoolWaveOp* poolWaveop)
{
    const EngineId engineId = poolWaveop->gEngineId();
    int numPrevs = 0;

    for (auto prevWaveEdge : poolWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveop->gType() == poolWaveop->gType()) {
            Assert(false, "Predecessors of Pool cannot be another Pool: ",
                prevWaveop->gName(), ", type: ", prevWaveop->gTypeStr());
        } else {
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }

    }
}


/***************************************************************
***************************************************************/
void
EventMgr::processActivation(wave::ActivationWaveOp* activationWaveop)
{
    const EngineId engineId = activationWaveop->gEngineId();
    int numPrevs = 0;

    for (auto prevWaveEdge : activationWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveop->gType() == activationWaveop->gType()) {
            Assert(false, "Predecessors of Activation cannot be another Activation: ",
                prevWaveop->gName(), ", type: ", prevWaveop->gTypeStr());
        } else {
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }
    }
}


void
EventMgr::
processSbAtomLoad(wave::SbAtomLoadWaveOp* sbatomLoadWaveop)
{
    const EngineId engineId = sbatomLoadWaveop->gEngineId();
    int numPrevs = 0;

    for (auto prevWaveEdge : sbatomLoadWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveop->gType() == sbatomLoadWaveop->gType()) {
            Assert(false, "Predecessors of SbAtomLoad cannot be another SbAtomLoad: ",
                prevWaveop->gName(), ", type: ", prevWaveop->gTypeStr());
        } else {
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }
    }
}


void
EventMgr::processSbAtomSave(wave::SbAtomSaveWaveOp* sbatomSaveWaveop)
{
    const EngineId engineId = sbatomSaveWaveop->gEngineId();
    int numPrevs = 0;

    for (auto prevWaveEdge : sbatomSaveWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveop->gType() == sbatomSaveWaveop->gType()) {
            Assert(false, "Predecessors of SbAtomSave cannot be another SbAtomSave: ",
                prevWaveop->gName(), ", type: ", prevWaveop->gTypeStr());
        } else {
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }
    }
}


void
EventMgr::processResAdd(wave::ResAddWaveOp* resaddWaveop)
{
    const EngineId engineId = resaddWaveop->gEngineId();
    int numPrevs = 0;

    for (auto prevWaveEdge : sbatomSaveWaveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveop->gType() == resaddWaveop->gType()) {
            Assert(false, "Predecessors of ResAdd cannot be another ResAdd: ",
                prevWaveop->gName(), ", type: ", prevWaveop->gTypeStr());
        } else {
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }
    }
}
#endif


void
EventMgr::processWaveop(wave::WaveOp* waveop)
{
    const EngineId engineId = waveop->gEngineId();
    int numPrevs = 0;

    for (auto prevWaveEdge : waveop->gPrevWaveEdges()) {
        auto prevWaveop = prevWaveEdge->gFromOp();
        if (prevWaveop->gEngineId() == engineId) {
            continue; // when two waveops execute on the same engine, no need for sync
        }
        ++numPrevs;

        if (prevWaveop->gType() == waveop->gType()) {
            Assert(false, "Predecessors of ", waveop->gTypeStr(), " cannot be another waveop of the same type: ",
                prevWaveop->gName());
        } else {
            const EventId eventId = getLocalEventId(prevWaveEdge);
            prevWaveEdge->rEvent(EventSetMode::OnEndWrDst, eventId, EventWaitMode::SetThenClear);
        }
    }
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
        } else {
            processWaveop(waveOp);
        }
#if 0
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
#endif
    }
}


/***************************************************************
***************************************************************/



} // namespace events


}


#include <limits>

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
    const EventId eventId = m_EventId++;
    if (m_EventId >= arch::Arch::gNumberTpbEvents()) {
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
    }
}


/***************************************************************
***************************************************************/



} // namespace events


}


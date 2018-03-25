#include "utils/inc/asserter.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/waveop.hpp"


namespace kcc {
namespace wave {


WaveEdge::WaveEdge(WaveOp* fromOp, WaveOp* toOp)
    : m_FromOp(fromOp)
    , m_ToOp(toOp)
{
}

void
WaveEdge::rEvent(events::EventSetMode setMode, events::EventId eventId, events::EventWaitMode waitMode)
{
    m_Channel.rEvent(setMode, eventId, waitMode);
}

bool
WaveEdge::qNeedToImplementWait() const
{
    if (this->gEventId() == events::EventId_Invalid) {
        Assert(! qNeedToWaitFor(), "Invalid event ID on an edge that need be waited for: from waveop '",
            gFromOp()->gName(), "', to waveop '", gToOp()->gName(), "'");
        return false;
    } else {
        Assert(qNeedToWaitFor(), "Valid event ID on an edge that need not be waited for: from waveop '",
            gFromOp()->gName(), "', to waveop '", gToOp()->gName(), "'");
    }
    return true;
}

bool
WaveEdge::qNeedToWaitFor() const
{
    const wave::WaveOp* const prevWaveop = gFromOp();
    const wave::WaveOp* const succWaveop = gToOp();

    // when two waveops execute on different engines, need for sync
    if (prevWaveop->gEngineId() != succWaveop->gEngineId()) {
        return true;
    }
    // when two waveops execute on the same engine, no need for sync
    // except for DMA. If Load depends on a Save, Load must wait on Save
    if (EngineId::DmaEng == prevWaveop->gEngineId()) {
        return true;
    }
    return false;
}


} // namespace wave
} // namespace kcc

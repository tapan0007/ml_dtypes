#include "wave/inc/waveedge.hpp"


namespace kcc {
namespace wave {


WaveEdge::WaveEdge(WaveOp* fromOp, WaveOp* toOp)
    : m_FromOp(fromOp)
    , m_ToOp(toOp)
{
}

void
WaveEdge::rEvent(events::EventSetMode setMode, EventId eventId, events::EventWaitMode waitMode)
{
    m_Channel.rEvent(setMode, eventId, waitMode);
}


} // namespace wave
} // namespace kcc

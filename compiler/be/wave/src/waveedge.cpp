#include "wave/inc/waveedge.hpp"


namespace kcc {
namespace wave {


WaveEdge::WaveEdge(WaveOp* fromOp, WaveOp* toOp)
    : m_FromOp(fromOp)
    , m_ToOp(toOp)
{
}

void
WaveEdge::rEvent(EventWaitMode waitMode, EventId eventId, EventSetMode setMode)
{
    m_Channel.rEvent(waitMode, eventId, setMode);
}


} // namespace wave
} // namespace kcc

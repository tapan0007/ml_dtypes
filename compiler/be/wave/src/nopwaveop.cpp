
#include <sstream>



#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"

#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "layers/inc/layer.hpp"
#include "nets/inc/network.hpp"

#include "wave/inc/waveconsts.hpp"
#include "wave/inc/waveedge.hpp"
#include "wave/inc/nopwaveop.hpp"


// #define RETURN_ASSERT(x) return (x)
#define RETURN_ASSERT(x)  assert(x); return (x)


namespace kcc {
namespace wave {

NopWaveOp::NopWaveOp(const NopWaveOp::Params& params,
                             const std::vector<WaveOp*>& prevWaveOps,
                             EngineId engineId, events::EventId evtId)
    : WaveOp(params, prevWaveOps) // will add back edges
    , m_EngineId(engineId)
{
    if (prevWaveOps.size() > 0) {
        Assert(prevWaveOps.size() == 1,
            "NopWaveOps can have 0 or 1 previous WaveOps. It has ", prevWaveOps.size());
        Assert(m_PrevWaveEdges.size() == 1,
            "NopWaveOps can have 0 or 1 previous WaveEdges. It has ", prevWaveOps.size());
        wave::WaveEdge* prevWaveEdge = m_PrevWaveEdges[0];
        prevWaveEdge->rEvent(events::EventSetMode::OnEndWrDst, evtId, events::EventWaitMode::WaitThenClear);
    }
}

const std::string&
NopWaveOp::gLayerName() const
{
    const static std::string name("NopLayer");
    return  name;
}



#define RETURN_ASSERT(x)  assert(x); return (x)


bool
NopWaveOp::verify() const
{
    // Don't call WaveOp::verify() since the layer is nullptr
    if (m_Name == "") {
        RETURN_ASSERT(false);
    }
    if (m_Order < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}




bool
NopWaveOp::Params::verify() const
{
    // Don't call WaveOp::Params::verify() since the layer is nullptr
    if (m_WaveOpName == "") {
        RETURN_ASSERT(false);
    }
    if (m_Order < 0) {
        RETURN_ASSERT(false);
    }
    return true;
}

std::string
NopWaveOp::gTypeStrStatic()
{
    return WaveOpTypeStr_Nop;
}

}}


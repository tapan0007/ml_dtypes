#include <sstream>



#include "utils/inc/asserter.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/events.hpp"

#include "layers/inc/layer.hpp"
#include "wave/inc/waveop.hpp"
#include "nets/inc/network.hpp"



namespace kcc {
namespace wave {

//----------------------------------------------------------------
WaveOp::WaveOp (const WaveOp::Params& params,
        const std::vector<WaveOp*>& prevWaveOps)
    : m_Name(params.m_WaveOpName)
    // , m_OfmapDesc(params.m_OfmapDesc)
    , m_Order(params.m_Order)
    , m_Layer(params.m_Layer)
{
    assert(params.verify());
    for (auto prevWaveOp : prevWaveOps) {
        m_PrevWaveOps.push_back(prevWaveOp);
    }
}

//----------------------------------------------------------------
const std::string&
WaveOp::gLayerName () const
{
    return m_Layer->gName();
}

#if 0
#endif


bool
WaveOp::verify () const
{
    if (m_Name == "") {
        return false;
    }
    if (! m_Layer) {
        return false;
    }
    if (m_Order < 0) {
        return false;
    }
    return true;
}





bool
WaveOp::Params::verify() const
{
    if (m_WaveOpName == "") {
        return false;
    }
    if (! m_Layer) {
        return false;
    }
    return true;
}

void
WaveOp::addSuccWaveop(WaveOp* succWaveop)
{
    m_SuccWaveOps.push_back(succWaveop);
}

void
WaveOp::rWaitEvent(EventId eventId, EventWaitMode mode)
{
    m_Sync.m_WaitEventId = eventId;
    m_Sync.m_WaitEventMode = mode;
}

void
WaveOp::rSetEvent(EventId eventId, EventSetMode mode)
{
    m_Sync.m_SetEventId = eventId;
    m_Sync.m_SetEventMode = mode;
}





WaveOp::Events::Events()
    : m_SetEventId(-1)
    , m_SetEventMode(EventSetMode::NoEvent)
    , m_WaitEventId(-1)
    , m_WaitEventMode(EventWaitMode::NoEvent)
{
}

}} // namespace


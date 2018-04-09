
#include "tpb_isa.hpp"
#include "compisa/inc/compisacommon.hpp"
#include "events/inc/events.hpp"

namespace kcc {
namespace compisa {

void InitSync(TPB_CMD_SYNC& sync)
{
    sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
    sync.set_event_mode     = events::SET_EVENT_INVALID;
}

}}


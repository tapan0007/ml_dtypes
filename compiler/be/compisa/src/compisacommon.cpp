
//#include "runtime/isa/common/aws_tonga_isa_common.h"
#include "aws_tonga_isa_common.h"

#include "compisa/inc/compisacommon.hpp"
#include "events/inc/events.hpp"

namespace kcc {
namespace compisa {

void InitSync(TONGA_ISA_TPB_INST_EVENTS& sync)
{
    sync.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::Invalid);
    sync.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::Invalid);
}

}}


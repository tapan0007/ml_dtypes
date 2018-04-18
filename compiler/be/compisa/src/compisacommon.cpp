
//#include "runtime/isa/common/aws_tonga_isa_common.h"


#include "utils/inc/asserter.hpp"

#include "compisa/inc/compisacommon.hpp"
#include "events/inc/events.hpp"

namespace kcc {
namespace compisa {

constexpr kcc_int32 BYTES_PER_WORD = TONGA_ISA_TPB_INST_NBYTES / TONGA_ISA_TPB_INST_NWORDS;
static_assert((TONGA_ISA_TPB_INST_NBYTES % TONGA_ISA_TPB_INST_NWORDS) == 0,
                    "Default instruction size must be integral multiple of word size");



void
InitEvent(uint8_t& event_idx)
{
    event_idx = events::EventId_Invalid();
}

void
InitSync(TONGA_ISA_TPB_INST_EVENTS& inst_events)
{
    inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::Invalid);
    inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::Invalid);
    InitEvent(inst_events.wait_event_idx);
    InitEvent(inst_events.set_event_idx);
}

void
InitHeader (TONGA_ISA_TPB_INST_HEADER& header, TONGA_ISA_TPB_OPCODE opcode, uint8_t sz)
{
    header.opcode = opcode;
    Assert((sz % BYTES_PER_WORD) == 0, "Instruction size must be integral multiple of word size");
    header.inst_word_len = sz / BYTES_PER_WORD;
    header.debug_level = 0;
    header.sw_reserved = 0;
}

}}


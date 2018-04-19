
//#include "runtime/isa/common/aws_tonga_isa_common.h"


#include "utils/inc/asserter.hpp"

#include "events/inc/events.hpp"

#include "compisa/inc/compisacommon.hpp"

#include "compisa/inc/compisaactivation.hpp"
#include "compisa/inc/compisaclear.hpp"
#include "compisa/inc/compisacommon.hpp"
#include "compisa/inc/compisacopy.hpp"
#include "compisa/inc/compisaldweights.hpp"
#include "compisa/inc/compisamatmul.hpp"
#include "compisa/inc/compisamemset.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisapool.hpp"
#include "compisa/inc/compisareciprocal.hpp"
#include "compisa/inc/compisaset.hpp"
#include "compisa/inc/compisasimmemcpy.hpp"
#include "compisa/inc/compisasimrdnpy.hpp"
#include "compisa/inc/compisasimwrnpy.hpp"
#include "compisa/inc/compisatensorreduceop.hpp"
#include "compisa/inc/compisatensorscalarop.hpp"
#include "compisa/inc/compisatensorscalarptrop.hpp"
#include "compisa/inc/compisatensortensorop.hpp"
#include "compisa/inc/compisawait.hpp"
#include "compisa/inc/compisawrite.hpp"

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

void
AllInstructions()
{
    ActivationInstr         actInstr;
    ClearInstr              clearInstr;
    CopyInstr               copyInstr;
    LdWeightsInstr          ldweightsInstr;
    MatMulInstr             matmulInstr;
    MemSetInstr             memsetInstr;
    NopInstr                nopInstr;
    PoolInstr               poolInstr;
    ReciprocalInstr         reciprocalInstr;
    SetInstr                setInstr;
    SimMemCpyInstr          memcpyInstr;
    SimRdNpyInstr           rdnpyInstr;
    SimWrNpyInstr           wrnpyInstr;
    TensorReduceOpInstr     tensorredInstr;
    TensorScalarOpInstr     tensorscalarInstr;
    TensorScalarPtrOpInstr  tensorscalarptrInstr;
    TensorTensorOpInstr     tensortensorInstr;
    WaitInstr               waitInstr;
    WriteInstr              writeInstr;
}

}}


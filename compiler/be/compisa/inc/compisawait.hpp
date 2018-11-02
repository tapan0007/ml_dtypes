#pragma once

#ifndef KCC_COMPISA_WAIT_H
#define KCC_COMPISA_WAIT_H



#include "aws_tonga_isa_tpb_event_wait.h"


#include "utils/inc/types.hpp"
#include "compisa/inc/compisacommon.hpp"


namespace kcc {

namespace compisa {

using WaitInstr = InstrTempl<::TONGA_ISA_TPB_EVENT_WAIT_INST,
                             ::TONGA_ISA_TPB_OPCODE_EVENT_WAIT,
                             ::tonga_isa_tpb_eventwait_check_validity>;


}}

#endif // KCC_COMPISA_WAIT_H


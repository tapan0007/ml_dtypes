#pragma once

#ifndef KCC_COMPISA_CLEAR_H
#define KCC_COMPISA_CLEAR_H



#include "aws_tonga_isa_tpb_event_clear.h"


#include "utils/inc/types.hpp"
#include "compisa/inc/compisacommon.hpp"


namespace kcc {

namespace compisa {

using ClearInstr = InstrTempl<TONGA_ISA_TPB_EVENT_CLEAR_INST,
                               TONGA_ISA_TPB_OPCODE_EVENT_CLEAR,
                               tonga_isa_tpb_eventclear_check_validity>;


}}

#endif // KCC_COMPISA_CLEAR_H


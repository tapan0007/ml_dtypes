#pragma once

#ifndef KCC_COMPISA_SET_H
#define KCC_COMPISA_SET_H



#include "aws_tonga_isa_tpb_event_set.h"


#include "utils/inc/types.hpp"
#include "compisa/inc/compisacommon.hpp"


namespace kcc {

namespace compisa {

using SetInstr = InstrTempl<TONGA_ISA_TPB_EVENT_SET_INST,
                            TONGA_ISA_TPB_OPCODE_EVENT_SET,
                            tonga_isa_tpb_eventset_check_validity>;


}}

#endif // KCC_COMPISA_SET_H


#pragma once

#ifndef KCC_COMPISA_NOP_H
#define KCC_COMPISA_NOP_H



#include "aws_tonga_isa_tpb_nop.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {

using NopInstr = InstrTempl<TONGA_ISA_TPB_NOP_INST,
                             TONGA_ISA_TPB_OPCODE_NOP,
                             tonga_isa_tpb_nop_check_validity>;


}}

#endif // KCC_COMPISA_NOP_H



#pragma once

#ifndef KCC_COMPISA_ACTIVATE_H
#define KCC_COMPISA_ACTIVATE_H



#include "aws_tonga_isa_tpb_activate.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {

using ActivateInstr = InstrTempl<TONGA_ISA_TPB_ACTIVATE_INST,
                                 TONGA_ISA_TPB_OPCODE_ACTIVATE,
                                 tonga_isa_tpb_activate_check_validity>;

}}

#endif // KCC_COMPISA_ACTIVATE_H


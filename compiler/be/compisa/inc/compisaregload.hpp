#pragma once

#ifndef KCC_COMPISA_REG_LOAD_H
#define KCC_COMPISA_REG_LOAD_H



#include "aws_tonga_isa_tpb_reg_load.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using RegLoadInstr = InstrTempl<::TONGA_ISA_TPB_REG_LOAD_INST,
                                ::TONGA_ISA_TPB_OPCODE_REG_LOAD,
                                ::tonga_isa_tpb_regload_check_validity>;

}}

#endif // KCC_COMPISA_REG_LOAD_H



#pragma once

#ifndef KCC_COMPISA_REG_SHUFFLE_H
#define KCC_COMPISA_REG_SHUFFLE_H



#include "aws_tonga_isa_tpb_reg_shuffle.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using RegShuffleInstr = InstrTempl<::TONGA_ISA_TPB_REG_SHUFFLE_INST,
                                   ::TONGA_ISA_TPB_OPCODE_REG_SHUFFLE,
                                   ::tonga_isa_tpb_regshuffle_check_validity>;


}}

#endif // KCC_COMPISA_REG_SHUFFLE_H



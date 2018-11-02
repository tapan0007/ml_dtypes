#pragma once

#ifndef KCC_COMPISA_RECIPROCAL_H
#define KCC_COMPISA_RECIPROCAL_H



#include "aws_tonga_isa_tpb_reciprocal.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using ReciprocalInstr = InstrTempl<::TONGA_ISA_TPB_RECIPROCAL_INST,
                                   ::TONGA_ISA_TPB_OPCODE_RECIPROCAL,
                                   ::tonga_isa_tpb_reciprocal_check_validity>;


}}

#endif // KCC_COMPISA_RECIPROCAL_H



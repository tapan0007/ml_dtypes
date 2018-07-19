#pragma once

#ifndef KCC_COMPISA_CAST_H
#define KCC_COMPISA_CAST_H



#include "aws_tonga_isa_tpb_cast.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using CastInstr = InstrTempl<TONGA_ISA_TPB_CAST_INST,
                             TONGA_ISA_TPB_OPCODE_CAST,
                             tonga_isa_tpb_cast_check_validity>;


}}

#endif // KCC_COMPISA_CAST_H



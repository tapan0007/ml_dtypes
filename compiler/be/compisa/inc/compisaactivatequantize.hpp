#pragma once

#ifndef KCC_COMPISA_ACTIVATE_QUANTIZE_H
#define KCC_COMPISA_ACTIVATE_QUANTIZE_H



#include "aws_tonga_isa_tpb_activate_quantize.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using ActivateQuantizeInstr = InstrTempl<TONGA_ISA_TPB_ACTIVATE_QUANTIZE_INST,
                             TONGA_ISA_TPB_OPCODE_ACTIVATE_QUANTIZE,
                             tonga_isa_tpb_activate_quantize_check_validity>;


}}

#endif // KCC_COMPISA_ACTIVATE_QUANTIZE_H


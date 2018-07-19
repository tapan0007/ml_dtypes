#pragma once

#ifndef KCC_COMPISA_COPY_H
#define KCC_COMPISA_COPY_H



#include "aws_tonga_isa_tpb_copy.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using CopyInstr = InstrTempl<TONGA_ISA_TPB_COPY_INST,
                             TONGA_ISA_TPB_OPCODE_COPY,
                             tonga_isa_tpb_copy_check_validity>;


}}

#endif // KCC_COMPISA_COPY_H



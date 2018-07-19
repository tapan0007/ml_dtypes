#pragma once

#ifndef KCC_COMPISA_MEMSET_H
#define KCC_COMPISA_MEMSET_H



#include "aws_tonga_isa_tpb_memset.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {
namespace compisa {

using MemSetInstr = InstrTempl<TONGA_ISA_TPB_MEMSET_INST,
                             TONGA_ISA_TPB_OPCODE_MEMSET,
                             tonga_isa_tpb_memset_check_validity>;



}}

#endif // KCC_COMPISA_MEMSET_H



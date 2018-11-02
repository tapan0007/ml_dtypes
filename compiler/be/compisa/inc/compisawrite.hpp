#pragma once

#ifndef KCC_COMPISA_WRITE_H
#define KCC_COMPISA_WRITE_H


#include "aws_tonga_isa_tpb_write.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {

using WriteInstr = InstrTempl<::TONGA_ISA_TPB_WRITE_INST,
                              ::TONGA_ISA_TPB_OPCODE_WRITE,
                              ::tonga_isa_tpb_write_check_validity>;


}}

#endif // KCC_COMPISA_WRITE_H


#pragma once

#ifndef KCC_COMPISA_SIMMEMCPY_H
#define KCC_COMPISA_SIMMEMCPY_H



#include "aws_tonga_isa_tpb_sim_memcpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {

inline TONGA_ISA_ERROR_CODE tonga_isa_tpb_sim_memcpy_check_validity(const TONGA_ISA_TPB_SIM_MEMCPY_INST*)
{
    return TONGA_ISA_ERR_CODE_SUCCESS;
}

using SimMemCpyInstr = InstrTempl<TONGA_ISA_TPB_SIM_MEMCPY_INST,
                             TONGA_ISA_TPB_OPCODE_SIM_MEMCPY,
                             tonga_isa_tpb_sim_memcpy_check_validity>;

}}

#endif // KCC_COMPISA_SIMMEMCPY_H


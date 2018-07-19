#pragma once

#ifndef KCC_COMPISA_SIM_RDNPY_H
#define KCC_COMPISA_SIM_RDNPY_H



#include "aws_tonga_isa_tpb_sim_rdnpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {

inline void tonga_isa_tpb_sim_rdnpy_check_validity(const TONGA_ISA_TPB_SIM_RDNPY_INST*)
{
}

using SimRdNpyInstr = InstrTempl<TONGA_ISA_TPB_SIM_RDNPY_INST,
                             TONGA_ISA_TPB_OPCODE_SIM_RDNPY,
                             tonga_isa_tpb_sim_rdnpy_check_validity>;


}}

#endif // KCC_COMPISA_SIM_RDNPY_H


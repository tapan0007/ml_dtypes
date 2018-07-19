#pragma once

#ifndef KCC_COMPISA_SIM_WRNPY_H
#define KCC_COMPISA_SIM_WRNPY_H



#include "aws_tonga_isa_tpb_sim_wrnpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {

inline void tonga_isa_tpb_sim_wrnpy_check_validity(const TONGA_ISA_TPB_SIM_WRNPY_INST*)
{
}

using SimWrNpyInstr = InstrTempl<TONGA_ISA_TPB_SIM_WRNPY_INST,
                             TONGA_ISA_TPB_OPCODE_SIM_WRNPY,
                             tonga_isa_tpb_sim_wrnpy_check_validity>;

}}

#endif // KCC_COMPISA_SIM_WRNPY_H


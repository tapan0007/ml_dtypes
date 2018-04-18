#pragma once

#ifndef KCC_COMPISA_SIMMEMCPY_H
#define KCC_COMPISA_SIMMEMCPY_H



#include "aws_tonga_isa_tpb_sim_memcpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SimMemCpyInstr : public TONGA_ISA_TPB_SIM_MEMCPY_INST {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SimMemCpyInstr()
        : TONGA_ISA_TPB_SIM_MEMCPY_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_SIM_MEMCPY);
    }

    void CheckValidity()
    {}
};


}}

#endif // KCC_COMPISA_SIMMEMCPY_H


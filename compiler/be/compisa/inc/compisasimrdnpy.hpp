#pragma once

#ifndef KCC_COMPISA_SIM_RDNPY_H
#define KCC_COMPISA_SIM_RDNPY_H



#include "aws_tonga_isa_tpb_sim_rdnpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SimRdNpyInstr : public TONGA_ISA_TPB_SIM_RDNPY_INST {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SimRdNpyInstr()
        : TONGA_ISA_TPB_SIM_RDNPY_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_SIM_RDNPY);
    }

    void CheckValidity() const
    {}
};


}}

#endif // KCC_COMPISA_SIM_RDNPY_H


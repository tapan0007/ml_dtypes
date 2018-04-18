#pragma once

#ifndef KCC_COMPISA_LDWEIGHTS_H
#define KCC_COMPISA_LDWEIGHTS_H



#include "aws_tonga_isa_tpb_ldweights.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class LdWeightsInstr : public TONGA_ISA_TPB_LDWEIGHTS_INST {
public:
    static constexpr EngineId engineId = EngineId::PeArray;
public:
    //----------------------------------------------------------------
    LdWeightsInstr()
        : TONGA_ISA_TPB_LDWEIGHTS_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_LDWEIGHTS);
    }

    void CheckValidity()
    {
        tonga_isa_tpb_ldweights_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_LDWEIGHTS_H


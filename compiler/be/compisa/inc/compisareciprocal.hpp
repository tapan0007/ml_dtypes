#pragma once

#ifndef KCC_COMPISA_RECIPROCAL_H
#define KCC_COMPISA_RECIPROCAL_H



#include "aws_tonga_isa_tpb_reciprocal.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class ReciprocalInstr : public TONGA_ISA_TPB_RECIPROCAL_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    ReciprocalInstr()
        : TONGA_ISA_TPB_RECIPROCAL_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_RECIPROCAL);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_reciprocal_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_RECIPROCAL_H



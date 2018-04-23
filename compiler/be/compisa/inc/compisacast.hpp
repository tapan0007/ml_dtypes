#pragma once

#ifndef KCC_COMPISA_CAST_H
#define KCC_COMPISA_CAST_H



#include "aws_tonga_isa_tpb_cast.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class CastInstr : public TONGA_ISA_TPB_CAST_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    CastInstr()
        : TONGA_ISA_TPB_CAST_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_CAST);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_cast_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_CAST_H



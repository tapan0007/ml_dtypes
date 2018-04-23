#pragma once

#ifndef KCC_COMPISA_REG_SHUFFLE_H
#define KCC_COMPISA_REG_SHUFFLE_H



#include "aws_tonga_isa_tpb_reg_shuffle.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class RegShuffleInstr : public TONGA_ISA_TPB_REG_SHUFFLE_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    RegShuffleInstr()
        : TONGA_ISA_TPB_REG_SHUFFLE_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_REG_SHUFFLE);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_regshuffle_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_REG_SHUFFLE_H



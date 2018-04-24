#pragma once

#ifndef KCC_COMPISA_NOP_H
#define KCC_COMPISA_NOP_H



#include "aws_tonga_isa_tpb_nop.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class NopInstr : public TONGA_ISA_TPB_NOP_INST {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    NopInstr()
        : TONGA_ISA_TPB_NOP_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_NOP);
        cycle_cnt = 0;
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_nop_check_validity(this);
    }

};


}}

#endif // KCC_COMPISA_NOP_H



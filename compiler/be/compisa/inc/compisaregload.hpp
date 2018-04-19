#pragma once

#ifndef KCC_COMPISA_REG_LOAD_H
#define KCC_COMPISA_REG_LOAD_H



#include "aws_tonga_isa_tpb_reg_load.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class RegLoadInstr : public TONGA_ISA_TPB_REG_LOAD_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    RegLoadInstr()
        : TONGA_ISA_TPB_REG_LOAD_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_REG_LOAD);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_regload_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_REG_LOAD_H



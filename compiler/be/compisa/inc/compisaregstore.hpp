#pragma once

#ifndef KCC_COMPISA_REG_STORE_H
#define KCC_COMPISA_REG_STORE_H



#include "aws_tonga_isa_tpb_reg_store.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class RegStoreInstr : public TONGA_ISA_TPB_REG_STORE_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    RegStoreInstr()
        : TONGA_ISA_TPB_REG_STORE_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_REG_STORE);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_regstore_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_REG_STORE_H



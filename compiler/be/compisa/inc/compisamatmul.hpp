#pragma once

#ifndef KCC_COMPISA_MATMUL_H
#define KCC_COMPISA_MATMUL_H



#include "aws_tonga_isa_tpb_matmul.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {


class MatMulInstr : public TONGA_ISA_TPB_MATMUL_INST {
public:
    static constexpr EngineId engineId = EngineId::PeArray;
public:
    //----------------------------------------------------------------
    MatMulInstr()
        : TONGA_ISA_TPB_MATMUL_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_MATMUL);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_matmul_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_MATMUL_H



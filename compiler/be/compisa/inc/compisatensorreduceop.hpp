#pragma once

#ifndef KCC_COMPISA_TENSORREDUCE_OP_H
#define KCC_COMPISA_TENSORREDUCE_OP_H



#include "aws_tonga_isa_tpb_tensor_reduce_op.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {

namespace compisa {


class TensorReduceOpInstr : public TONGA_ISA_TPB_TENSOR_REDUCE_OP_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    TensorReduceOpInstr()
        : TONGA_ISA_TPB_TENSOR_REDUCE_OP_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_TENSOR_REDUCE_OP);
    }

	void CheckValidity() const
    {
        tonga_isa_tpb_tensorreduce_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_TENSORREDUCE_OP_H



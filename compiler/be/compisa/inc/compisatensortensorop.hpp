#pragma once

#ifndef KCC_COMPISA_TENSORTENSOROP_H
#define KCC_COMPISA_TENSORTENSOROP_H



#include "aws_tonga_isa_tpb_tensor_tensor_op.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {

namespace compisa {


class TensorTensorOpInstr : public TONGA_ISA_TPB_TENSOR_TENSOR_OP_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    TensorTensorOpInstr()
        : TONGA_ISA_TPB_TENSOR_TENSOR_OP_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_TENSOR_TENSOR_OP);
    }

	void CheckValidity() const
    {
		tonga_isa_tpb_tensortensor_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_TENSORTENSOROP_H



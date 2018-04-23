#pragma once

#ifndef KCC_COMPISA_TENSORSCALAROP_H
#define KCC_COMPISA_TENSORSCALAROP_H



#include "aws_tonga_isa_tpb_tensor_scalar_op.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {

namespace compisa {


class TensorScalarOpInstr : public TONGA_ISA_TPB_TENSOR_SCALAR_OP_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    TensorScalarOpInstr()
        : TONGA_ISA_TPB_TENSOR_SCALAR_OP_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_TENSOR_SCALAR_OP);
    }

	void CheckValidity() const
    {
        tonga_isa_tpb_tensorscalar_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_TENSORSCALAROP_H



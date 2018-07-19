#pragma once

#ifndef KCC_COMPISA_TENSORREDUCE_OP_H
#define KCC_COMPISA_TENSORREDUCE_OP_H



#include "aws_tonga_isa_tpb_tensor_reduce_op.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {
namespace compisa {

using TensorReduceOpInstr = InstrTempl<TONGA_ISA_TPB_TENSOR_REDUCE_OP_INST,
                             TONGA_ISA_TPB_OPCODE_TENSOR_REDUCE_OP,
                             tonga_isa_tpb_tensorreduce_check_validity>;


}}

#endif // KCC_COMPISA_TENSORREDUCE_OP_H



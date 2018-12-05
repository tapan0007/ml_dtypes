#pragma once

#ifndef KCC_COMPISA_TENSORSCALAROP_H
#define KCC_COMPISA_TENSORSCALAROP_H



#include "aws_tonga_isa_tpb_tensor_scalar_op.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {
namespace compisa {

using TensorScalarOpInstr = InstrTempl2<
    ::TONGA_ISA_TPB_TENSOR_SCALAR_OP_INST,
    ::tonga_isa_tpb_tensorscalar_check_validity>;


}}

#endif // KCC_COMPISA_TENSORSCALAROP_H



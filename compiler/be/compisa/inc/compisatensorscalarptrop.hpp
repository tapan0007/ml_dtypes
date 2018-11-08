#pragma once

#ifndef KCC_COMPISA_TENSORSCALARPTROP_H
#define KCC_COMPISA_TENSORSCALARPTROP_H



#include "aws_tonga_isa_tpb_tensor_scalar_ptr_op.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {
namespace compisa {

using TensorScalarPtrOpInstr = InstrTempl2<
    ::TONGA_ISA_TPB_TENSOR_SCALAR_PTR_OP_INST,
    ::tonga_isa_tpb_tensorscalarptr_check_validity>;

}}

#endif // KCC_COMPISA_TENSORSCALARPTROP_H



#pragma once

#ifndef KCC_COMPISA_MATMUL_H
#define KCC_COMPISA_MATMUL_H



#include "aws_tonga_isa_tpb_matmul.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {
namespace compisa {


using MatMulInstrBase = InstrTempl<::TONGA_ISA_TPB_MATMUL_INST,
                                   ::TONGA_ISA_TPB_OPCODE_MATMUL,
                                   ::tonga_isa_tpb_matmul_check_validity>;

class MatMulInstr : public MatMulInstrBase {
public:
    bool qAsynchrnous() const {
        return true;
    }
};

}}

#endif // KCC_COMPISA_MATMUL_H



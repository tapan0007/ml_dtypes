#pragma once

#ifndef KCC_COMPISA_WRITE_H
#define KCC_COMPISA_WRITE_H


#include "aws_tonga_isa_tpb_write.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class WriteInstr : public TONGA_ISA_TPB_WRITE_INST {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    WriteInstr()
        : TONGA_ISA_TPB_WRITE_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_WRITE);
    }

    void CheckValidity() {
        tonga_isa_tpb_write_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_WRITE_H


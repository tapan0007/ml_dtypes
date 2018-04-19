#pragma once

#ifndef KCC_COMPISA_COPY_H
#define KCC_COMPISA_COPY_H



#include "aws_tonga_isa_tpb_copy.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class CopyInstr : public TONGA_ISA_TPB_COPY_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    CopyInstr()
        : TONGA_ISA_TPB_COPY_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_COPY);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_copy_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_COPY_H



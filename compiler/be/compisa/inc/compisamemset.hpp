#pragma once

#ifndef KCC_COMPISA_MEMSET_H
#define KCC_COMPISA_MEMSET_H



#include "aws_tonga_isa_tpb_memset.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class MemSetInstr : public TONGA_ISA_TPB_MEMSET_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    MemSetInstr()
        : TONGA_ISA_TPB_MEMSET_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_MEMSET);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_memset_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_MEMSET_H



#pragma once

#ifndef KCC_COMPISA_ACTIVATE_H
#define KCC_COMPISA_ACTIVATE_H



#include "aws_tonga_isa_tpb_activate.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class ActivateInstr : public TONGA_ISA_TPB_ACTIVATE_INST {
public:
    static constexpr EngineId engineId = EngineId::Activation;
public:
    //----------------------------------------------------------------
    ActivateInstr()
        : TONGA_ISA_TPB_ACTIVATE_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_ACTIVATE);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_activate_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_ACTIVATE_H


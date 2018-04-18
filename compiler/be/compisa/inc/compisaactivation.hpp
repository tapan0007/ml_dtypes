#pragma once

#ifndef KCC_COMPISA_ACTIVATION_H
#define KCC_COMPISA_ACTIVATION_H



#include "aws_tonga_isa_tpb_activate.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class ActivationInstr : public TONGA_ISA_TPB_ACTIVATE_INST {
public:
    static constexpr EngineId engineId = EngineId::Activation;
public:
    //----------------------------------------------------------------
    ActivationInstr()
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

#endif // KCC_COMPISA_ACTIVATION_H


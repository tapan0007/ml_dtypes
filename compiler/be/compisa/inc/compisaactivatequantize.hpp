#pragma once

#ifndef KCC_COMPISA_ACTIVATE_QUANTIZE_H
#define KCC_COMPISA_ACTIVATE_QUANTIZE_H



#include "aws_tonga_isa_tpb_activate_quantize.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class ActivateQuantizeInstr : public TONGA_ISA_TPB_ACTIVATE_QUANTIZE_INST {
public:
    static constexpr EngineId engineId = EngineId::Activation;
public:
    //----------------------------------------------------------------
    ActivateQuantizeInstr()
        : TONGA_ISA_TPB_ACTIVATE_QUANTIZE_INST()
    {
        InitInstructionWithEmbEvent(*this, TONGA_ISA_TPB_OPCODE_ACTIVATE);
    }

    void CheckValidity() const
    {
        tonga_isa_tpb_activate_quantize_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_ACTIVATE_QUANTIZE_H


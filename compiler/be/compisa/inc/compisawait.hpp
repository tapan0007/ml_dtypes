#pragma once

#ifndef KCC_COMPISA_WAIT_H
#define KCC_COMPISA_WAIT_H



#include "aws_tonga_isa_tpb_event_wait.h"


#include "utils/inc/types.hpp"
#include "compisa/inc/compisacommon.hpp"


namespace kcc {

namespace compisa {


class WaitInstr : public TONGA_ISA_TPB_EVENT_WAIT_INST {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    WaitInstr()
        : TONGA_ISA_TPB_EVENT_WAIT_INST()
    {
        InitEventInstruction(*this, TONGA_ISA_TPB_OPCODE_EVENT_WAIT);
    }

    void CheckValidity()
    {
        tonga_isa_tpb_eventwait_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_WAIT_H


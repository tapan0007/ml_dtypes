#pragma once

#ifndef KCC_COMPISA_SET_H
#define KCC_COMPISA_SET_H



#include "aws_tonga_isa_tpb_event_set.h"


#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SetInstr : public TONGA_ISA_TPB_EVENT_SET_INST {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SetInstr()
        : TONGA_ISA_TPB_EVENT_SET_INST()
    {
        InitEventInstruction(*this, TONGA_ISA_TPB_OPCODE_EVENT_SET);
    }

    void CheckValidity()
    {
        tonga_isa_tpb_eventset_check_validity(this);
    }
};


}}

#endif // KCC_COMPISA_SET_H


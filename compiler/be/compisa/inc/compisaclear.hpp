#pragma once

#ifndef KCC_COMPISA_CLEAR_H
#define KCC_COMPISA_CLEAR_H



#include "aws_tonga_isa_tpb_event_clear.h"


#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class ClearInstr : public TONGA_ISA_TPB_EVENT_CLEAR_INST {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    ClearInstr()
        : TONGA_ISA_TPB_EVENT_CLEAR_INST()
    {
    }

};


}}

#endif // KCC_COMPISA_CLEAR_H


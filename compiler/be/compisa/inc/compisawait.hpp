#pragma once

#ifndef KCC_COMPISA_WAIT_H
#define KCC_COMPISA_WAIT_H



#include "shared/inc/tpb_isa_wait.hpp"


#include "utils/inc/consts.hpp"


namespace kcc {

namespace compisa {


class WaitInstr : public WAIT {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    WaitInstr()
        : WAIT()
    {
    }

};


}}

#endif // KCC_COMPISA_WAIT_H


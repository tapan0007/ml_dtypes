#pragma once

#ifndef KCC_COMPISA_CLEAR_H
#define KCC_COMPISA_CLEAR_H



#include "shared/inc/tpb_isa_clear.hpp"


#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class ClearInstr : public CLEAR {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    ClearInstr()
        : CLEAR()
    {
    }

};


}}

#endif // KCC_COMPISA_CLEAR_H


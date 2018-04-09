#pragma once

#ifndef KCC_COMPISA_MATADD_H
#define KCC_COMPISA_MATADD_H



#include "shared/inc/tpb_isa_matadd.hpp"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"

namespace kcc {

namespace compisa {


class MatAddInstr : public MATADD {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    MatAddInstr()
        : MATADD()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_MATADD_H


#pragma once

#ifndef KCC_COMPISA_POOL_H
#define KCC_COMPISA_POOL_H



#include "shared/inc/tpb_isa_pool.hpp"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class PoolInstr : public POOL {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    PoolInstr()
        : POOL()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_POOL_H


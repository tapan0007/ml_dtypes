#pragma once

#ifndef KCC_COMPISA_POOL_H
#define KCC_COMPISA_POOL_H



#include "aws_tonga_isa_tpb_pool.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class PoolInstr : public TONGA_ISA_TPB_POOL_INST {
public:
    static constexpr EngineId engineId = EngineId::Pooling;
public:
    //----------------------------------------------------------------
    PoolInstr()
        : TONGA_ISA_TPB_POOL_INST()
    {
        InitSync(inst_events);
    }

};


}}

#endif // KCC_COMPISA_POOL_H


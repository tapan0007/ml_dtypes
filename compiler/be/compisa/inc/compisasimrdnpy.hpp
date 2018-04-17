#pragma once

#ifndef KCC_COMPISA_SIM_RDNPY_H
#define KCC_COMPISA_SIM_RDNPY_H



#include "aws_tonga_isa_tpb_sim_rdnpy.h"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SimRdNpyInstr : public SIM_RDNPY {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SimRdNpyInstr()
        : SIM_RDNPY()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_SIM_RDNPY_H


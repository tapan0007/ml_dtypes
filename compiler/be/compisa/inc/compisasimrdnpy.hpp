#pragma once

#ifndef KCC_COMPISA_SIM_RDNPY_H
#define KCC_COMPISA_SIM_RDNPY_H



#include "shared/inc/tpb_isa_simrdnpy.hpp"

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


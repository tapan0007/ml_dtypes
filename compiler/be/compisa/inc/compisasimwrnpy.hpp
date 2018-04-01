#pragma once

#ifndef KCC_COMPISA_SIM_WRNPY_H
#define KCC_COMPISA_SIM_WRNPY_H



#include "shared/inc/tpb_isa_simwrnpy.hpp"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SimWrNpyInstr : public SIM_WRNPY {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SimWrNpyInstr()
        : SIM_WRNPY()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_SIM_WRNPY_H


#pragma once

#ifndef KCC_COMPISA_SIMMEMCPY_H
#define KCC_COMPISA_SIMMEMCPY_H



#include "shared/inc/tpb_isa_simmemcpy.hpp"

#include "compisa/inc/compisacommon.hpp"
#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class SimMemCpyInstr : public SIM_MEMCPY {
public:
    static constexpr EngineId engineId = EngineId::DmaEng;
public:
    //----------------------------------------------------------------
    SimMemCpyInstr()
        : SIM_MEMCPY()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_SIMMEMCPY_H


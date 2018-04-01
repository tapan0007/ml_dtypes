#pragma once

#ifndef KCC_COMPISA_LDWEIGHTS_H
#define KCC_COMPISA_LDWEIGHTS_H



#include "shared/inc/tpb_isa_ldweights.hpp"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class LdWeightsInstr : public LDWEIGHTS {
public:
    static constexpr EngineId engineId = EngineId::PeArray;
public:
    //----------------------------------------------------------------
    LdWeightsInstr()
        : LDWEIGHTS()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_LDWEIGHTS_H


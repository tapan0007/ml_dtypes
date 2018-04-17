#pragma once

#ifndef KCC_COMPISA_LDWEIGHTS_H
#define KCC_COMPISA_LDWEIGHTS_H



#include "aws_tonga_isa_tpb_ldweights.h"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"


namespace kcc {

namespace compisa {


class LdWeightsInstr : public TONGA_ISA_TPB_LDWEIGHTS_INST {
public:
    static constexpr EngineId engineId = EngineId::PeArray;
public:
    //----------------------------------------------------------------
    LdWeightsInstr()
        : TONGA_ISA_TPB_LDWEIGHTS_INST()
    {
        InitSync(inst_events);
    }

};


}}

#endif // KCC_COMPISA_LDWEIGHTS_H


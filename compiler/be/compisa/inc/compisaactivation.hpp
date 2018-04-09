#pragma once

#ifndef KCC_COMPISA_ACTIVATION_H
#define KCC_COMPISA_ACTIVATION_H



#include "shared/inc/tpb_isa_activate.hpp"

#include "compisa/inc/compisacommon.hpp"

#include "utils/inc/types.hpp"



namespace kcc {

namespace compisa {


class ActivationInstr : public ACTIVATION {
public:
    static constexpr EngineId engineId = EngineId::Activation;
public:
    //----------------------------------------------------------------
    ActivationInstr()
        : ACTIVATION()
    {
        InitSync(sync);
    }

};


}}

#endif // KCC_COMPISA_ACTIVATION_H


#pragma once

#ifndef KCC_COMPISA_LDWEIGHTS_H
#define KCC_COMPISA_LDWEIGHTS_H



#include "shared/inc/tpb_isa_ldweights.hpp"

#include "utils/inc/consts.hpp"

#include "events/inc/events.hpp"

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
        // override from Inkling
        sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
        sync.set_event_mode     = events::SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_LDWEIGHTS_H


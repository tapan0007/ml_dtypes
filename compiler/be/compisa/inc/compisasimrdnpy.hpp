#pragma once

#ifndef KCC_COMPISA_SIM_RDNPY_H
#define KCC_COMPISA_SIM_RDNPY_H



#include "shared/inc/tpb_isa_simrdnpy.hpp"

#include "events/inc/events.hpp"

namespace kcc {

namespace compisa {


class SimRdNpyInstr : public SIM_RDNPY {
public:
    //----------------------------------------------------------------
    SimRdNpyInstr()
        : SIM_RDNPY()
    {
        // override from Inkling
        sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
        sync.set_event_mode     = events::SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_SIM_RDNPY_H


#pragma once

#ifndef KCC_COMPISA_SIMMEMCPY_H
#define KCC_COMPISA_SIMMEMCPY_H



#include "shared/inc/tpb_isa_simmemcpy.hpp"

#include "events/inc/events.hpp"

namespace kcc {

namespace compisa {


class SimMemCpyInstr : public SIM_MEMCPY {
public:
    //----------------------------------------------------------------
    SimMemCpyInstr()
        : SIM_MEMCPY()
    {
        // override from Inkling
        sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
        sync.set_event_mode     = events::SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_SIMMEMCPY_H


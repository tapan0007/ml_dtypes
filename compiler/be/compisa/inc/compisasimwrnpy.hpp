#pragma once

#ifndef KCC_COMPISA_SIM_WRNPY_H
#define KCC_COMPISA_SIM_WRNPY_H



#include "shared/inc/tpb_isa_simwrnpy.hpp"


namespace kcc {

namespace compisa {


class SimWrNpyInstr : public SIM_WRNPY {
public:
    //----------------------------------------------------------------
    SimWrNpyInstr()
        : SIM_WRNPY()
    {
        // override from Inkling
        sync.wait_event_mode    = WAIT_EVENT_INVALID;
        sync.set_event_mode     = SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_SIM_WRNPY_H


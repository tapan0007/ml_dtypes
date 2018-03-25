#pragma once

#ifndef KCC_COMPISA_ACTIVATION_H
#define KCC_COMPISA_ACTIVATION_H



#include "shared/inc/tpb_isa_activate.hpp"


namespace kcc {

namespace compisa {


class ActivationInstr : public ACTIVATION {
public:
    //----------------------------------------------------------------
    ActivationInstr()
        : ACTIVATION()
    {
        // override from Inkling
        sync.wait_event_mode    = WAIT_EVENT_INVALID;
        sync.set_event_mode     = SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_ACTIVATION_H


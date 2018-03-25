#pragma once

#ifndef KCC_COMPISA_LDWEIGHTS_H
#define KCC_COMPISA_LDWEIGHTS_H



#include "shared/inc/tpb_isa_ldweights.hpp"


namespace kcc {

namespace compisa {


class LdWeightsInstr : public LDWEIGHTS {
public:
    //----------------------------------------------------------------
    LdWeightsInstr()
        : LDWEIGHTS()
    {
        // override from Inkling
        sync.wait_event_mode    = WAIT_EVENT_INVALID;
        sync.set_event_mode     = SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_LDWEIGHTS_H


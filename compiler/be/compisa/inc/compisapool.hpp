#pragma once

#ifndef KCC_COMPISA_POOL_H
#define KCC_COMPISA_POOL_H



#include "shared/inc/tpb_isa_pool.hpp"

#include "events/inc/events.hpp"

namespace kcc {

namespace compisa {


class PoolInstr : public POOL {
public:
    //----------------------------------------------------------------
    PoolInstr()
        : POOL()
    {
        // override from Inkling
        sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
        sync.set_event_mode     = events::SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_POOL_H


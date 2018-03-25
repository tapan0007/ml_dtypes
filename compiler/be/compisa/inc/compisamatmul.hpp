#pragma once

#ifndef KCC_COMPISA_MATMUL_H
#define KCC_COMPISA_MATMUL_H



#include "shared/inc/tpb_isa_matmul.hpp"

#include "events/inc/events.hpp"

namespace kcc {

namespace compisa {


class MatMulInstr : public MATMUL {
public:
    //----------------------------------------------------------------
    MatMulInstr()
        : MATMUL()
    {
        // override from Inkling
        sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
        sync.set_event_mode     = events::SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_MATMUL_H



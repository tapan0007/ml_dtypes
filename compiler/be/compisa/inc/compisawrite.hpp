#pragma once

#ifndef KCC_COMPISA_WRITE_H
#define KCC_COMPISA_WRITE_H



#include "shared/inc/tpb_isa_write.hpp"


namespace kcc {

namespace compisa {


class WriteInstr : public WRITE {
public:
    //----------------------------------------------------------------
    WriteInstr()
        : WRITE()
    {
        // override from Inkling
        sync.wait_event_mode    = WAIT_EVENT_INVALID;
        sync.set_event_mode     = SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_WRITE_H


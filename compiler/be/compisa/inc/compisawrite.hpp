#pragma once

#ifndef KCC_COMPISA_WRITE_H
#define KCC_COMPISA_WRITE_H



#include "shared/inc/tpb_isa_write.hpp"

#include "utils/inc/consts.hpp"

#include "events/inc/events.hpp"

namespace kcc {

namespace compisa {


class WriteInstr : public WRITE {
public:
    static constexpr EngineId engineId = EngineId::AnyEng;
public:
    //----------------------------------------------------------------
    WriteInstr()
        : WRITE()
    {
        // override from Inkling
        sync.wait_event_mode    = events::WAIT_EVENT_INVALID;
        sync.set_event_mode     = events::SET_EVENT_INVALID;
    }

};


}}

#endif // KCC_COMPISA_WRITE_H


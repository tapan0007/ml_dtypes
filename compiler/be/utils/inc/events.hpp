#pragma once

#ifndef KCC_UTILS_EVENTS_H
#define KCC_UTILS_EVENTS_H 1

#include "shared/inc/tpb_isa.hpp"

namespace kcc {

enum class EventWaitMode {
    NoEvent         = NO_WAIT_EVENT,
    SetOnly         = WAIT_EVENT_SET,
    SetThenClear    = WAIT_EVENT_SET_THEN_CLEAR,
};

enum class EventSetMode {
    NoEvent         = NO_SET_EVENT,
    OnEndRdSrc      = SET_EVENT_ON_END_RD_SRC,
    OnEndWrDst      = SET_EVENT_ON_END_WR_DST,
};

int eventWaitMode2Int(EventWaitMode mode);
int eventSetMode2Int(EventSetMode mode);

}

#endif


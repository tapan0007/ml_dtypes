#pragma once

#ifndef KCC_EVENTS_EVENTS_H
#define KCC_EVENTS_EVENTS_H 1

#include "shared/inc/tpb_isa.hpp"

#include "utils/inc/types.hpp"

namespace kcc {
namespace events {

/****************************************************************
 *                                                              *
 ****************************************************************/
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





/****************************************************************
 *                                                              *
 ****************************************************************/





/****************************************************************
 *                                                              *
 ****************************************************************/
class Channel {
public:
    Channel();

    void rEvent(EventSetMode setMode, EventId eventId, EventWaitMode waitMode);

    EventId gEventId() const {
        return m_EventId;
    }
    EventWaitMode gWaitEventMode() const {
        return m_WaitEventMode;
    }
    EventSetMode gSetEventMode() const {
        return m_SetEventMode;
    }

private:
    EventSetMode        m_SetEventMode;
    EventId             m_EventId;
    EventWaitMode       m_WaitEventMode;
};


/****************************************************************
 *                                                              *
 ****************************************************************/
int eventWaitMode2Int(EventWaitMode mode);
int eventSetMode2Int(EventSetMode mode);

}}

#endif


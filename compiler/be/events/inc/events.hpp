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
class WaitEvent {
public:
    WaitEvent();

    void rEvent(EventId eventId, EventWaitMode);

    EventId gId() const {
        return m_EventId;
    }

    EventWaitMode gMode() const {
        return m_EventMode;
    }

    static int eventWaitMode2Int(EventWaitMode mode);

private:
    WaitEvent(const WaitEvent&) = delete;

private:
    EventId             m_EventId;
    EventWaitMode       m_EventMode;
};




/****************************************************************
 *                                                              *
 ****************************************************************/
class SetEvent {
public:
    SetEvent();

    void rEvent(EventId eventId, EventSetMode mode);

    EventId gId() const {
        return m_EventId;
    }

    EventSetMode gMode() const {
        return m_EventMode;
    }

    static int eventSetMode2Int(EventSetMode mode);


private:
    SetEvent(const SetEvent&) = delete;

private:
    EventId             m_EventId;
    EventSetMode        m_EventMode;
};




/****************************************************************
 *                                                              *
 ****************************************************************/
class Events {
public:
    Events();

    void rWaitEvent(EventId eventId, EventWaitMode);
    void rSetEvent(EventId eventId, EventSetMode);

    EventId gWaitEventId() const {
        return m_WaitEvent.gId();
    }
    EventWaitMode gWaitEventMode() const {
        return m_WaitEvent.gMode();
    }
    EventId gSetEventId() const {
        return m_SetEvent.gId();
    }
    EventSetMode gSetEventMode() const {
        return m_SetEvent.gMode();
    }

private:
    WaitEvent           m_WaitEvent;
    SetEvent            m_SetEvent;
};


/****************************************************************
 *                                                              *
 ****************************************************************/
int eventWaitMode2Int(EventWaitMode mode);
int eventSetMode2Int(EventSetMode mode);

}}

#endif


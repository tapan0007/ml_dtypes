#pragma once

#ifndef KCC_EVENTS_EVENTS_H
#define KCC_EVENTS_EVENTS_H 1

#include "shared/inc/tpb_isa.hpp"

#include "utils/inc/types.hpp"

namespace kcc {
namespace events {

//**********************************************************************
using EventId = kcc_int32;

static constexpr EventId EventId_Invalid()
{
    return NUM_TPB_EVENTS - 1;
}


/****************************************************************
 *                                                              *
 ****************************************************************/
enum class EventWaitMode {
    NoEvent         = NO_WAIT_EVENT,
    SetOnly         = WAIT_EVENT_SET,
    SetThenClear    = WAIT_EVENT_SET_THEN_CLEAR,

    Invalid         = WAIT_EVENT_INVALID
};

enum class EventSetMode {
    NoEvent         = NO_SET_EVENT,
    OnEndRdSrc      = SET_EVENT_ON_END_RD_SRC,
    OnEndWrDst      = SET_EVENT_ON_END_WR_DST,

    Invalid         = SET_EVENT_INVALID
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
    Channel(const Channel&) = delete;
    Channel& operator= (const Channel&) = delete;

private:
    EventSetMode        m_SetEventMode  = EventSetMode::Invalid;
    EventId             m_EventId       = EventId_Invalid();
    EventWaitMode       m_WaitEventMode = EventWaitMode::Invalid;
};


/****************************************************************
 *                                                              *
 ****************************************************************/
int eventWaitMode2Int(EventWaitMode mode);
int eventSetMode2Int(EventSetMode mode);

}}

#endif


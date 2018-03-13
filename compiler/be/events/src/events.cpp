
#include "events/inc/events.hpp"


namespace kcc {
namespace events {

/****************************************************************
 *                                                              *
 ****************************************************************/
Events::Events()
    : m_WaitEvent()
    , m_SetEvent()
{
}


void
Events::rWaitEvent(EventId eventId, EventWaitMode mode)
{
    m_WaitEvent.rEvent(eventId, mode);
}


void
Events::rSetEvent(EventId eventId, EventSetMode mode)
{
    m_SetEvent.rEvent(eventId, mode);
}






/****************************************************************
 *                                                              *
 ****************************************************************/
WaitEvent::WaitEvent()
    : m_EventId(-1)
    , m_EventMode(EventWaitMode::NoEvent)
{
}

void
WaitEvent::rEvent(EventId eventId, EventWaitMode mode)
{
    m_EventId = eventId;
    m_EventMode = mode;
}


int
WaitEvent::eventWaitMode2Int(EventWaitMode mode)
{
    switch(mode) {
    case EventWaitMode::NoEvent:
        return NO_WAIT_EVENT;
        break;
    case EventWaitMode::SetOnly:
        return WAIT_EVENT_SET;
        break;
    case EventWaitMode::SetThenClear:
        return WAIT_EVENT_SET_THEN_CLEAR;
        break;
    }
    Assert(false, "Wrong EventWaitMode: ", static_cast<kcc_int32>(mode));
    return 0;
}



/****************************************************************
 *                                                              *
 ****************************************************************/
SetEvent::SetEvent()
    : m_EventId(-1)
    , m_EventMode(EventSetMode::NoEvent)
{
}

void
SetEvent::rEvent(EventId eventId, EventSetMode mode)
{
    m_EventId = eventId;
    m_EventMode = mode;
}

int
SetEvent::eventSetMode2Int(EventSetMode mode)
{
    switch(mode) {
    case EventSetMode::NoEvent:
        return NO_SET_EVENT;
        break;
    case EventSetMode::OnEndRdSrc:
        return SET_EVENT_ON_END_RD_SRC;
        break;
    case EventSetMode::OnEndWrDst:
        return SET_EVENT_ON_END_WR_DST;
        break;
    }
    Assert(false, "Wrong EventSetMode: ", static_cast<kcc_int32>(mode));
    return 0;
}

}} // namespace


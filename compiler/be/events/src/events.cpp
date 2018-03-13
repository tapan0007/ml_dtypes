
#include "utils/inc/asserter.hpp"

#include "events/inc/events.hpp"


namespace kcc {
namespace events {

/****************************************************************
 *                                                              *
 ****************************************************************/
Channel::Channel()
    : m_WaitEventMode(EventWaitMode::NoEvent)
    , m_EventId(-1)
    , m_SetEventMode(EventSetMode::NoEvent)
{ }


void
Channel::rEvent(EventWaitMode waitMode, EventId eventId, EventSetMode setMode)
{
    m_WaitEventMode = waitMode;
    m_EventId = eventId;
    m_SetEventMode = setMode;
}





/****************************************************************
 *                                                              *
 ****************************************************************/
int
eventWaitMode2Int(EventWaitMode mode)
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
int
eventSetMode2Int(EventSetMode mode)
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


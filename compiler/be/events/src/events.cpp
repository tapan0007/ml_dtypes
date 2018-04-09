
#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "events/inc/events.hpp"


namespace kcc {
namespace events {

EventId EventId_Invalid()
{
    return arch::Arch::gArch().gNumberAllTpbEvents() - 1;
}

/****************************************************************
 *                                                              *
 ****************************************************************/
Channel::Channel()
    : m_SetEventMode(EventSetMode::DontSet)
    , m_EventId(EventId_Invalid())
    , m_WaitEventMode(EventWaitMode::DontWait)
{ }


void
Channel::rEvent(EventSetMode setMode, EventId eventId, EventWaitMode waitMode)
{
    m_SetEventMode = setMode;  // FromOp
    m_EventId = eventId;
    m_WaitEventMode = waitMode; // ToOp
}


bool qEventSetModeValid(kcc_int32 mode)
{
    return NO_WAIT_EVENT==mode || WAIT_EVENT_SET==mode || WAIT_EVENT_SET_THEN_CLEAR==mode;
}

bool qEventWaitModeValid(kcc_int32 mode)
{
    return NO_SET_EVENT==mode || SET_EVENT_ON_END_RD_SRC==mode || SET_EVENT_ON_END_WR_DST==mode;
}


/****************************************************************
 *                                                              *
 ****************************************************************/
int
eventWaitMode2Int(EventWaitMode mode)
{
    switch(mode) {
    case EventWaitMode::DontWait:
        return NO_WAIT_EVENT;
        break;
    case EventWaitMode::WaitOnly:
        return WAIT_EVENT_SET;
        break;
    case EventWaitMode::WaitThenClear:
        return WAIT_EVENT_SET_THEN_CLEAR;
        break;
    case EventWaitMode::Invalid:
        return events::WAIT_EVENT_INVALID;
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
    case EventSetMode::DontSet:
        return NO_SET_EVENT;
        break;
    case EventSetMode::OnEndRdSrc:
        return SET_EVENT_ON_END_RD_SRC;
        break;
    case EventSetMode::OnEndWrDst:
        return SET_EVENT_ON_END_WR_DST;
        break;
    case EventSetMode::Invalid:
        return events::SET_EVENT_INVALID;
        break;
    }
    Assert(false, "Wrong EventSetMode: ", static_cast<kcc_int32>(mode));
    return 0;
}

}} // namespace


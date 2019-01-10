#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "events/inc/events.hpp"


namespace kcc {
namespace events {

EventId EventId_Invalid()
{
    return -1;
}

EventId EventId_LastNonReserved()
{
    return arch::Arch::gArch().gNumberAllTpbEvents() - 1; // 0xFF
}



/****************************************************************
 *                                                              *
 ****************************************************************/
EventChannel::EventChannel()
    : m_SetEventMode(EventSetMode::DontSet)
    , m_EventId(EventId_Invalid())
    , m_WaitEventMode(EventWaitMode::DontWait)
{ }


void
EventChannel::rEvent(EventSetMode setMode, EventId eventId, EventWaitMode waitMode)
{
    m_SetEventMode  = setMode;  // FromOp
    m_EventId       = eventId;
    m_WaitEventMode = waitMode; // ToOp
}

void
EventChannel::clear()
{
    m_SetEventMode  = EventSetMode::DontSet;
    m_EventId       = EventId_Invalid();
    m_WaitEventMode = EventWaitMode::DontWait;
}


bool qEventWaitModeValid(kcc_int32 mode)
{
    switch (mode) {
    case TONGA_ISA_TPB_MODE_WAIT_NONE:
    case TONGA_ISA_TPB_MODE_WAIT_FOR_SET:
    case TONGA_ISA_TPB_MODE_WAIT_FOR_SET_THEN_CLEAR:
        return true;
    default:
        return false;
    }
}

bool qEventSetModeValid(kcc_int32 mode)
{
    switch (mode) {
    case TONGA_ISA_TPB_MODE_SET_NONE:
    case TONGA_ISA_TPB_MODE_SET_ON_DONE_RD_SRC:
    case TONGA_ISA_TPB_MODE_SET_ON_DONE_WR_DST:
    case TONGA_ISA_TPB_MODE_SET_ON_INST_DONE:
        return true;
    default:
        return false;
    }
}


/****************************************************************
 *                                                              *
 ****************************************************************/
TONGA_ISA_TPB_WAIT_EVENT_MODE
eventWaitMode2Isa(EventWaitMode mode)
{
    switch(mode) {
    case EventWaitMode::DontWait:
        return TONGA_ISA_TPB_MODE_WAIT_NONE;
        break;
    case EventWaitMode::WaitOnly:
        return TONGA_ISA_TPB_MODE_WAIT_FOR_SET;
        break;
    case EventWaitMode::WaitThenClear:
        return TONGA_ISA_TPB_MODE_WAIT_FOR_SET_THEN_CLEAR;
        break;
    case EventWaitMode::Invalid:
        return TONGA_ISA_TPB_MODE_WAIT_INVALID;
        break;
    }
    Assert(false, "Wrong EventWaitMode: ", static_cast<kcc_int32>(mode));
    return TONGA_ISA_TPB_MODE_WAIT_INVALID;
}



/****************************************************************
 *                                                              *
 ****************************************************************/
TONGA_ISA_TPB_SET_EVENT_MODE
eventSetMode2Isa(EventSetMode mode)
{
    switch(mode) {
    case EventSetMode::DontSet:
        return TONGA_ISA_TPB_MODE_SET_NONE;
        break;
    case EventSetMode::OnEndRdSrc:
        return TONGA_ISA_TPB_MODE_SET_ON_DONE_RD_SRC;
        break;
    case EventSetMode::OnEndWrDst:
        return TONGA_ISA_TPB_MODE_SET_ON_DONE_WR_DST;
        break;
    case EventSetMode::OnEndInstr:
        return TONGA_ISA_TPB_MODE_SET_ON_INST_DONE;
        break;
    case EventSetMode::Invalid:
        return TONGA_ISA_TPB_MODE_SET_INVALID;
        break;
    }
    Assert(false, "Wrong EventSetMode: ", static_cast<kcc_int32>(mode));
    return TONGA_ISA_TPB_MODE_SET_INVALID;
}

}} // namespace


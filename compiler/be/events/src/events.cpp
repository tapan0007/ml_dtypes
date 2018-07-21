#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "events/inc/events.hpp"


namespace kcc {
namespace events {

EventId EventId_Invalid()
{
    return -1;
}

EventId EventId_StartInference()
{
    return arch::Arch::gArch().gNumberAllTpbEvents() - 1;  // 0xFF
}

EventId EventId_BeforeInputRead_PeArray()
{
    return  EventId_StartInference() - 1;
}

EventId EventId_BeforeInputRead_ActEng()
{
    return EventId_BeforeInputRead_PeArray() - 1;
}


// kaena-531: There's only 1 delay from MM to following event set instr when there are
// multiple SETs (multiple dependencies), so to properly trigger a dependent load,
// there must be an event from MM to a WAIT followed by the first SETs (no longer embedded)
// followed by the next series of SETs. Reusing the last event ID (255) since that
// was used only for the start of inference.
EventId EventId_MMStartMultiSet()
{
    return EventId_BeforeInputRead_ActEng() - 1;
}

EventId EventId_LastNonReserved()
{
    return EventId_MMStartMultiSet() - 1;
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
    switch (mode) {
    case TONGA_ISA_TPB_MODE_WAIT_NONE:
    case TONGA_ISA_TPB_MODE_WAIT_FOR_SET:
    case TONGA_ISA_TPB_MODE_WAIT_FOR_SET_THEN_CLEAR:
        return true;
    default:
        return false;
    }
}

bool qEventWaitModeValid(kcc_int32 mode)
{
    switch (mode) {
    case TONGA_ISA_TPB_MODE_WAIT_NONE:
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


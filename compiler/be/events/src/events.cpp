#include "uarch_cfg.hpp"

#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "events/inc/events.hpp"

//Event_t
static_assert(NUM_TPB_EVENTS <= 
             (1U << 8*sizeof(TONGA_ISA_TPB_INST_EVENTS::wait_event_idx)),
    "Number of TPB events (NUM_TPB_EVENTS) too large for type Event_t");


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
    return TONGA_ISA_TPB_MODE_WAIT_NONE==mode || TONGA_ISA_TPB_MODE_WAIT_FOR_SET==mode || TONGA_ISA_TPB_MODE_WAIT_FOR_SET_THEN_CLEAR==mode;
}

bool qEventWaitModeValid(kcc_int32 mode)
{
    return TONGA_ISA_TPB_MODE_WAIT_NONE==mode || TONGA_ISA_TPB_MODE_SET_ON_DONE_RD_SRC==mode || TONGA_ISA_TPB_MODE_SET_ON_DONE_WR_DST==mode;
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
    case EventSetMode::Invalid:
        return TONGA_ISA_TPB_MODE_SET_INVALID;
        break;
    }
    Assert(false, "Wrong EventSetMode: ", static_cast<kcc_int32>(mode));
    return TONGA_ISA_TPB_MODE_SET_INVALID;
}

}} // namespace


#pragma once

#ifndef KCC_EVENTS_EVENTS_H
#define KCC_EVENTS_EVENTS_H 1

#include "shared/inc/uarch_cfg.hpp"
#include "shared/inc/tpb_isa.hpp"

#include "arch/inc/arch.hpp"

//Event_t
static_assert(NUM_TPB_EVENTS <= (1U << 8*sizeof(TPB_CMD_SYNC::wait_event_id)),
    "Number of TPB events (NUM_TPB_EVENTS) too large for type Event_t");


#include "utils/inc/types.hpp"

namespace kcc {
namespace events {

//**********************************************************************
using EventId = kcc_int32;

constexpr EventId EventId_Invalid()
{
    return arch::Arch::gNumberTpbEvents() - 1;
}

constexpr kcc_int32 KccMax3(kcc_int32 a, kcc_int32 b, kcc_int32 c)
{
    return a > b ? (a > c ? a : c)
                 : (b > c ? b : c);
}
                   

enum {
    SET_EVENT_INVALID = 1 + std::max(NO_SET_EVENT,
                                std::max(SET_EVENT_ON_END_WR_DST, SET_EVENT_ON_END_RD_SRC)),

    WAIT_EVENT_INVALID = 1 + std::max(NO_WAIT_EVENT,
                                std::max(WAIT_EVENT_SET, WAIT_EVENT_SET_THEN_CLEAR))
};

/****************************************************************
 *                                                              *
 ****************************************************************/
enum class EventWaitMode {
    DontWait        = NO_WAIT_EVENT,
    WaitOnly        = WAIT_EVENT_SET,
    WaitThenClear   = WAIT_EVENT_SET_THEN_CLEAR,

    Invalid         = WAIT_EVENT_INVALID
};

enum class EventSetMode {
    DontSet         = NO_SET_EVENT,
    OnEndRdSrc      = SET_EVENT_ON_END_RD_SRC,
    OnEndWrDst      = SET_EVENT_ON_END_WR_DST,

    Invalid         = SET_EVENT_INVALID
};

bool qEventWaitModeValid(kcc_int32 mode);
bool qEventSetModeValid(kcc_int32 mode);





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


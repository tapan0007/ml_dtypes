#pragma once

#ifndef KCC_EVENTS_EVENTS_H
#define KCC_EVENTS_EVENTS_H 1

//#include "shared/inc/uarch_cfg.hpp"
//#include "shared/inc/tpb_isa.hpp"

#include "aws_tonga_isa_tpb_common.h"

#include "arch/inc/arch.hpp"

//Event_t

#include "utils/inc/types.hpp"

namespace kcc {
namespace events {

//**********************************************************************
using EventId = kcc_int32;

EventId EventId_Invalid();

#if 0
constexpr kcc_int32 KccMax3(kcc_int32 a, kcc_int32 b, kcc_int32 c)
{
    return a > b ? (a > c ? a : c)
                 : (b > c ? b : c);
}
                   

enum {
    SET_EVENT_INVALID = 1 + std::max(TONGA_ISA_TPB_MODE_SET_NONE,
                                std::max(TONGA_ISA_TPB_MODE_SET_ON_DONE_WR_DST,
                                    TONGA_ISA_TPB_MODE_SET_ON_DONE_RD_SRC)),

    WAIT_EVENT_INVALID = 1 + std::max(TONGA_ISA_TPB_MODE_WAIT_NONE,
                                std::max(TONGA_ISA_TPB_MODE_WAIT_FOR_SET, 
                                    TONGA_ISA_TPB_MODE_WAIT_FOR_SET_THEN_CLEAR))
};
#endif

/****************************************************************
 *                                                              *
 ****************************************************************/
enum class EventWaitMode {
    DontWait        = TONGA_ISA_TPB_MODE_WAIT_NONE,
    WaitOnly        = TONGA_ISA_TPB_MODE_WAIT_FOR_SET,
    WaitThenClear   = TONGA_ISA_TPB_MODE_WAIT_FOR_SET_THEN_CLEAR,

    Invalid         = TONGA_ISA_TPB_MODE_WAIT_INVALID
};

enum class EventSetMode {
    DontSet         = TONGA_ISA_TPB_MODE_WAIT_NONE,
    OnEndRdSrc      = TONGA_ISA_TPB_MODE_SET_ON_DONE_RD_SRC,
    OnEndWrDst      = TONGA_ISA_TPB_MODE_SET_ON_DONE_WR_DST,

    Invalid         = TONGA_ISA_TPB_MODE_SET_INVALID
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
TONGA_ISA_TPB_WAIT_EVENT_MODE eventWaitMode2Isa(EventWaitMode mode);
TONGA_ISA_TPB_SET_EVENT_MODE  eventSetMode2Isa(EventSetMode mode);

}}

#endif


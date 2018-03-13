#pragma once

#ifndef KCC_WAVE_WAVEEDGE_H
#define KCC_WAVE_WAVEEDGE_H


#include <string>
#include <vector>
#include <assert.h>



#include "shared/inc/tpb_isa.hpp"


#include "utils/inc/types.hpp"
#include "utils/inc/consts.hpp"
#include "utils/inc/datatype.hpp"
#include "utils/inc/fmapdesc.hpp"
#include "events/inc/events.hpp"


namespace kcc {
enum class EventWaitMode;
enum class EventSetMode;

namespace layers {
    class Layer;
}
namespace nets {
    class Network;
}

namespace wave {

class WaveOp;

using namespace utils;

//--------------------------------------------------------
// The base class of all wave.
//--------------------------------------------------------
class WaveEdge {
protected:

    //----------------------------------------------------
public:
    //----------------------------------------------------------------
    WaveEdge(WaveOp* fromOp, WaveOp* toOp);

private:
    WaveEdge() = delete;
    WaveEdge(const WaveEdge&) = delete;
    WaveEdge& operator= (const WaveEdge&) const = delete;


public:

    const WaveOp* gFromOp() const {
        return m_FromOp;
    }

    WaveOp* gFromOp() {
        return m_FromOp;
    }

    const WaveOp* gToOp() const {
        return m_ToOp;
    }

    WaveOp* gToOp() {
        return m_ToOp;
    }

    /****************************************************************
     *                                                              *
     ****************************************************************/
    void rEvent(events::EventSetMode setMode, EventId eventId, events::EventWaitMode waitMode)
    {
        m_Channel.rEvent(setMode, eventId, waitMode);
    }
    EventId gEventId() const {
        return m_Channel.gEventId();
    }
    events::EventWaitMode gWaitEventMode() const {
        return m_Channel.gWaitEventMode();
    }
    events::EventSetMode gSetEventMode() const {
        return m_Channel.gSetEventMode();
    }

private:
    WaveOp*                 m_FromOp;
    WaveOp*                 m_ToOp;
    events::Channel         m_Channel;
}; // class WaveEdge



} // namespace wave
} // namespace kcc

#endif // KCC_WAVE_WAVEEDGE_H


#pragma once

#ifndef KCC_WAVE_WAVEEDGE_H
#define KCC_WAVE_WAVEEDGE_H


#include <string>
#include <vector>
#include <assert.h>





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
public:
public:
    enum class SyncMethod {
        None,
        WithEvent,
        WithSemaphore,
    };

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
    void rEvent(events::EventSetMode setMode, events::EventId eventId, events::EventWaitMode waitMode);
    void clearEvent();

    events::EventId gEventId() const {
        return m_EventChannel.gEventId();
    }
    events::EventWaitMode gWaitEventMode() const {
        return m_EventChannel.gWaitEventMode();
    }
    events::EventSetMode gSetEventMode() const {
        return m_EventChannel.gSetEventMode();
    }

    bool qNeedToImplementSync() const;
    bool qNeedToSync() const;
    bool qCanSyncWithSemaphore() const;

    bool qSyncedWithEvent() const;
    bool qSyncedWithSemaphore() const;
    void DoSyncWithSemaphore() {
        m_SyncMethod = SyncMethod::WithSemaphore;
    }

    bool qChosenForSuccSbAtom() const {
        return m_ChosenForSuccSbAtom;
    }
    void rChosenForSuccSbAtom(bool chosen) {
        m_ChosenForSuccSbAtom = chosen;
    }

private:
    void rSyncMethod(SyncMethod method) {
        m_SyncMethod = method;
    }
private:
    WaveOp*                 m_FromOp;
    WaveOp*                 m_ToOp;
    events::EventChannel    m_EventChannel;
    bool                    m_ChosenForSuccSbAtom = false;
    SyncMethod              m_SyncMethod = SyncMethod::None;
}; // class WaveEdge



} // namespace wave
} // namespace kcc

#endif // KCC_WAVE_WAVEEDGE_H


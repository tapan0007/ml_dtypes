#pragma once

#include <set>


#include "shared/inc/uarch_cfg.hpp"

#ifndef KCC_EVENTS_EVENTMGR_H
#define KCC_EVENTS_EVENTMGR_H

namespace kcc {

namespace wave {
class MatMulWaveOp;
class PoolWaveOp;
class ActivationWaveOp;
class SbAtomLoadWaveOp;
class SbAtomSaveWaveOp;
class ResAddWaveOp;
}


namespace events {

class EventMgr {
public:
    EventMgr(const nets::Network& network);
    void processWaveops();

private:
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    enum BarrierEvent {
        BarrierEvent_FromPe,
        BarrierEvent_FromAct,
        BarrierEvent_FromPool,
        BarrierEvent_ToPe,
        BarrierEvent_ToAct,
        BarrierEvent_ToPool,
        BarrierEvent_FirstNonBarrierEvent,
    };

    void initEventSets();
    void assignEventsToNewSuccEdges(wave::WaveOp* waveop);

private:
    const nets::Network& m_Network;
    EventId m_EventId;

    std::set<EventId> m_Available;
    std::set<EventId> m_InFlight;
    std::set<EventId> m_Completed;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H


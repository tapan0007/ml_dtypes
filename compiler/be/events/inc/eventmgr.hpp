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
class BarrierWaveOp;
}


namespace events {

class EventMgr {
public:
    EventMgr(nets::Network& network);
    void processWaveops();

private:
    using EventSet = std::set<EventId>;
private:
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    enum BarrierEvent {
        BarrierEvent_FromPe,
        BarrierEvent_FromAct,
        BarrierEvent_FromPool,
        BarrierEvent_FromStreamProc,
        BarrierEvent_FromDma,

        BarrierEvent_ToPe,
        BarrierEvent_ToAct,
        BarrierEvent_ToPool,
        BarrierEvent_ToStreamProc,
        BarrierEvent_ToDma,

        BarrierEvent_FirstNonBarrierEvent,
    };

    void initEventSets();
    void assignEventsToNewSuccEdges(wave::WaveOp* waveop);
    void moveCompletedEventsToAvailable();
    void completeEventsOnPrevEdges(wave::WaveOp* waveop);

    EngineId gBarrierEngineId(const wave::WaveOp* prevWaveop, const wave::WaveOp* succWaveop);
    void findWaveopsOnOtherEngines(kcc_int32 waveopIdx, const EngineId barrierEngId, bool backward,
                                       std::vector<wave::WaveOp*>& prevWaveops);
    void insertBarriers();
    void assignEventsToBarrier(wave::BarrierWaveOp* barrierWaveop);

    static EventId gEventIdToBarrier(EngineId fromEngId);
    static EventId gEventIdFromBarrier(EngineId toEngId);

    static bool qBarrierEvent(EventId evtId) {
        return 0 <= evtId && evtId < BarrierEvent_FirstNonBarrierEvent;
    }
    static bool qEventRegular(EventId eventId) {
        return BarrierEvent_FirstNonBarrierEvent <= eventId && eventId < EventId_Invalid();
    }

    void mvFromInFlightToCompleted(EventId eventId);
    void mvFromAvailableToInFlight(EventId eventId);
    void mvFromCompletedToAvailable(EventId eventId);
    void mvEventFromSetToSet(EventId evtId, EventSet& from, EventSet& to,
            const char* fromStr, const char* toStr);

    void verifyWaveop(const wave::WaveOp* waveop) const;

private:
    nets::Network& m_Network;
    EventId m_EventId;


    EventSet m_Available;
    EventSet m_InFlight;
    EventSet m_Completed;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H


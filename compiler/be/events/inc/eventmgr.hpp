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
    static kcc_int32 gNumberReservedTpbEvents() {
        return ReservedEvent_FirstNonReserved;
    }

private:
    using EventSet = std::set<EventId>;
private:
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    enum ReservedEvent {
        ReservedEvent_PeAct,
        ReservedEvent_PePool,
        ReservedEvent_PeDma,
        ReservedEvent_PeSp,

        ReservedEvent_ActPe,
        ReservedEvent_ActPool,
        ReservedEvent_ActDma,
        ReservedEvent_ActSp,

        ReservedEvent_PoolPe,
        ReservedEvent_PoolAct,
        ReservedEvent_PoolDma,
        ReservedEvent_PoolSp,

        ReservedEvent_DmaPe,
        ReservedEvent_DmaAct,
        ReservedEvent_DmaPool,
        ReservedEvent_DmaSp,

        ReservedEvent_SpPe,
        ReservedEvent_SpAct,
        ReservedEvent_SpPool,
        ReservedEvent_SpDma,

        ReservedEvent_FirstNonReserved
    };

    void initEventSets();
    void assignEventsToNewSuccEdges(wave::WaveOp* waveop);
    void moveCompletedEventsToAvailable();
    void completeEventsOnPrevEdges(wave::WaveOp* waveop);

    static EngineId gBarrierEngineId();
    void insertBarriers();

    static EventId gEventIdBetweenEngines(EngineId fromId, EngineId toId);

    static bool qReservedEvent(EventId evtId) {
        return 0 <= evtId && evtId < ReservedEvent_FirstNonReserved;
    }
    static bool qEventRegular(EventId eventId) {
        return ReservedEvent_FirstNonReserved <= eventId
               && eventId < EventId_Invalid();
    }


    wave::NopWaveOp* mkNopWaveop(wave::WaveOp* prevWaveop, EngineId engId, kcc_int32 waveopIdx);

    void mvFromInFlightToCompleted(EventId eventId);
    void mvFromAvailableToInFlight(EventId eventId);
    void mvFromCompletedToAvailable(EventId eventId);
    void mvEventFromSetToSet(EventId evtId, EventSet& from, EventSet& to,
            const char* fromStr, const char* toStr);

    void verifyWaveop(const wave::WaveOp* waveop) const;

private:
    nets::Network& m_Network;


    EventSet m_Available;
    EventSet m_InFlight;
    EventSet m_Completed;
    kcc_int32 m_NopIdx = 0;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H

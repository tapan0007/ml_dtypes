#pragma once

#include <set>

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
private:
    using EventSet = std::set<EventId>;
    class EventState {
    public:
        void clearAll();
        void clearCompleted();

        decltype(auto)
        addAvailable(EventId eventId) {
            return m_Available.insert(eventId);
        }

        bool availableEmpty() const {
            return m_Available.empty();
        }

        EventId gFirstAvailable() const {
            return *m_Available.begin();
        }

        void init();
        void mvFromAvailableToInFlight(EventId eventId);
        void moveCompletedEventsToAvailable();
        void mvFromInFlightToCompleted(EventId eventId);
        void mvFromCompletedToAvailable(EventId eventId);

        static void mvEventFromSetToSet(EventId evtId, EventSet& from, EventSet& to,
            const char* fromStr, const char* toStr);

        size_t gNumAvailable() const {
            return m_Available.size();
        }


    private:
        EventSet m_Available;
        EventSet m_InFlight;
        EventSet m_Completed;
    };

public:
    EventMgr(nets::Network& network);
    void processWaveops(bool kelf);
    static kcc_int32 gNumberReservedTpbEvents() {
        return ReservedEvent_FirstNonReserved;
    }

private:
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    enum ReservedEvent {
        ReservedEvent_PeAct,
        ReservedEvent_ActPool,
        ReservedEvent_PoolDma,

        ReservedEvent_DmaPool,
        ReservedEvent_PoolAct,
        ReservedEvent_ActPe,


        /*
        ReservedEvent_PePool,
        ReservedEvent_PeDma,
        ReservedEvent_PeSp,

        ReservedEvent_ActDma,
        ReservedEvent_ActSp,

        ReservedEvent_PoolPe,
        ReservedEvent_PoolSp,

        ReservedEvent_DmaPe,
        ReservedEvent_DmaAct,
        ReservedEvent_DmaSp,

        ReservedEvent_SpPe,
        ReservedEvent_SpAct,
        ReservedEvent_SpPool,
        ReservedEvent_SpDma,
        */

        ReservedEvent_FirstNonReserved
    };

    void assignEventsToNewSuccEdges(wave::WaveOp* waveop);
    void completeEventsOnPrevEdges(wave::WaveOp* waveop);

    void insertBarriers();

    static EventId gEventIdBetweenEngines(EngineId fromId, EngineId toId);

    static bool qReservedEvent(EventId evtId) {
        return 0 <= evtId && evtId < ReservedEvent_FirstNonReserved;
    }
    static bool qEventRegular(EventId eventId) {
        return ReservedEvent_FirstNonReserved <= eventId
               && eventId < EventId_StartInference();
    }


    wave::NopWaveOp* mkNopWaveop(wave::WaveOp* prevWaveop, EngineId engId, kcc_int32 waveopIdx);


    void verifyWaveop(const wave::WaveOp* waveop) const;

private:
    nets::Network& m_Network;


    EventState m_EventState;
    kcc_int32 m_NopIdx = 0;
    bool m_Kelf;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H


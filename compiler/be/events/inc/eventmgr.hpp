#pragma once

#include <set>
#include <map>

#ifndef KCC_EVENTS_EVENTMGR_H
#define KCC_EVENTS_EVENTMGR_H

namespace kcc {

namespace wave {
class MatMulWaveOp;
class PoolWaveOp;
class NopWaveOp;
class ActivationWaveOp;
class SbAtomLoadWaveOp;
class SbAtomSaveWaveOp;
class SbAtomWaveOp;
class DataMoveWaveOp;
}

namespace dma {
class DmaQueue;
}


namespace events {

class EventMgr {
private:
    using EventSet = std::set<EventId>;
    class EventState {
    public:
        EventState(const EventMgr& eventMgr)
            : m_EventMgr(eventMgr)
        {}

        EventState() = delete;

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

        void reset();
        void mvFromAvailableToInFlight(EventId eventId);
        void moveCompletedEventsToAvailable();
        void mvFromInFlightToCompleted(EventId eventId);
        void mvFromCompletedToAvailable(EventId eventId);

        void mvEventFromSetToSet(EventId evtId, EventSet& from, EventSet& to,
            const char* fromStr, const char* toStr);

        size_t gNumAvailable() const {
            return m_Available.size();
        }


    private:
        const EventMgr& m_EventMgr;
        EventSet m_Available;
        EventSet m_InFlight;
        EventSet m_Completed;
    };

public:
    EventMgr(nets::Network& network);
    ~EventMgr();

    void processWaveops(bool useSem);

    bool qKelf() const {
        return m_Kelf;
    }
    void rKelf(bool kelf) {
        m_Kelf = kelf;
    }

public:
    kcc_int32 gNumberReservedTpbEvents() const {
        return 1 + EventId_TpbLastReserved() - EventId_TpbFirstReserved();
    }
    static EventId EventId_TpbFirstReserved() {
        return ReservedEvent_BarrierFirst;
    }
    EventId EventId_TpbLastReserved() const {
        return m_Kelf ? ReservedEvent_BarrierLast_WithKelf : ReservedEvent_BarrierLast_WithAngel;
    }
    EventId EventId_FirstNonReserved() const {
        return EventId_TpbLastReserved() + 1;
    }

    static EventId EventId_MMStartMultiSet()
    {
        return ReservedEvent_MMStartMultiSet;
    }
    static EventId EventId_RunTimeFirst()
    {
        return ReservedEvent_RunTimeFirst;
    }
    static EventId EventId_RunTimeLast()
    {
        return ReservedEvent_RunTimeLast;
    }

private:
    enum  {
        RunTimeReservedEventsCount = 23,
        RunTimeReservedSemaphoresCount = 3,
    };

    enum ReservedEvent {
        ReservedEvent_RunTimeFirst = 0,
        ReservedEvent_RunTimeLast = ReservedEvent_RunTimeFirst + RunTimeReservedEventsCount - 1,

        // kaena-531: There's only 1 delay from MM to following event set instr when there are
        // multiple SETs (multiple dependencies), so to properly trigger a dependent load,
        // there must be an event from MM to a WAIT followed by the first SETs (no longer embedded)
        // followed by the next series of SETs.
        ReservedEvent_MMStartMultiSet,

        ReservedEvent_PeAct,
            ReservedEvent_BarrierFirst = ReservedEvent_PeAct,
        ReservedEvent_ActPool,
        ReservedEvent_PoolAct,
        ReservedEvent_ActPe,
            ReservedEvent_BarrierLast_WithKelf = ReservedEvent_ActPe,

        ReservedEvent_PoolAngel,
        ReservedEvent_AngelPool,
            ReservedEvent_BarrierLast_WithAngel = ReservedEvent_AngelPool,
    };

    enum ReservedSemaphore {
        ReservedSemaphore_RunTimeFirst = 0,
        ReservedSemaphore_RunTimeLast = ReservedSemaphore_RunTimeFirst + RunTimeReservedSemaphoresCount - 1,
        ReservedSemaphore_FirstNonReserved
    };



private:
    void assignEventsToNewSuccEdges(wave::WaveOp* waveop);
    void completeEventsOnPrevEdges(wave::WaveOp* waveop);

    void insertBarriers();
    void insertOneBarrier(kcc_int32 waveopIdx,
                          std::vector<wave::WaveOp*>& newWaveops);

    void determineQueuesAndSemaphoreValues();
    const dma::DmaQueue* findQueue(const wave::DataMoveWaveOp* sbatomWop, bool firstQueue);

    EventId gEventIdBetweenEngines(EngineId fromId, EngineId toId) const;

    bool qReservedEvent(EventId evtId) const {
        return 0 <= evtId && evtId < EventId_FirstNonReserved();
    }
    bool qEventRegular(EventId eventId) const {
        return EventId_FirstNonReserved() <= eventId
               && eventId <= EventId_LastNonReserved();
    }

    static EventSetMode gEventSetMode(const wave::WaveOp* waveop,
                                      EventSetMode setMode);

private:
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    void linkBarrierNops(std::vector<wave::WaveOp*>& newWaveops);

    wave::NopWaveOp* mkNopWaveop(wave::WaveOp* prevWaveop, EngineId engId, kcc_int32 waveopIdx);


    void verifyWaveop(const wave::WaveOp* waveop) const;

private:
    bool qUseSemaphore() const {
        return m_UseSemaphore;
    }
    bool qUseEventsOnly() const {
        return !qUseSemaphore();
    }
    bool qUseEvent(const wave::WaveEdge* edge) const;

private:
    nets::Network& m_Network;
    bool m_UseSemaphore = false;

    std::map<std::string, const dma::DmaQueue*> m_Name2Queue;
    std::map<const dma::DmaQueue*, kcc_int32> m_DmaQueueCount;


    EventState m_EventState;
    kcc_int32 m_NopIdx = 0;
    bool m_Kelf = true;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H


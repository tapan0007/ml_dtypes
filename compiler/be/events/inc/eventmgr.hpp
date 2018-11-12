#pragma once

#include <set>

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
class BarrierWaveOp;
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
    ~EventMgr();

    void processWaveops(bool kelf, bool useSem);
    static kcc_int32 gNumberReservedTpbEvents() {
        return ReservedEvent_FirstNonReserved;
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
    void processMatMult(wave::MatMulWaveOp* matmulWaveop);
    void processWaveop(wave::WaveOp* waveop);

    EventId getLocalEventId(const wave::WaveEdge* edge);

    enum  {
        RunTimeReservedEventsCount = 7,
        RunTimeReservedSemaphoresCount = 3,
    };
    enum ReservedEvent {
        ReservedEvent_RunTimeFirst = 0,
        ReservedEvent_RunTimeLast = ReservedEvent_RunTimeFirst + RunTimeReservedEventsCount - 1,


        ReservedEvent_PeAct,
        ReservedEvent_ActPool,
        ReservedEvent_PoolAngel,

        ReservedEvent_AngelPool,
        ReservedEvent_PoolAct,
        ReservedEvent_ActPe,

        // kaena-531: There's only 1 delay from MM to following event set instr when there are
        // multiple SETs (multiple dependencies), so to properly trigger a dependent load,
        // there must be an event from MM to a WAIT followed by the first SETs (no longer embedded)
        // followed by the next series of SETs.
        ReservedEvent_MMStartMultiSet,


        ReservedEvent_FirstNonReserved
    };
    enum ReservedSemaphore {
        ReservedSemaphore_RunTimeFirst = 0,
        ReservedSemaphore_RunTimeLast = ReservedSemaphore_RunTimeFirst + RunTimeReservedSemaphoresCount - 1,
        ReservedSemaphore_FirstNonReserved
    };

    void assignEventsToNewSuccEdges(wave::WaveOp* waveop);
    void completeEventsOnPrevEdges(wave::WaveOp* waveop);

    void insertBarriers();
    void insertOneBarrier(kcc_int32 waveopIdx,
                          std::vector<wave::WaveOp*>& newWaveops);

    void determineQueuesAndSemaphoreValues();
    const dma::DmaQueue* findQueue(const wave::SbAtomWaveOp* sbatomWop);

    static EventId gEventIdBetweenEngines(EngineId fromId, EngineId toId);

    static bool qReservedEvent(EventId evtId) {
        return 0 <= evtId && evtId < ReservedEvent_FirstNonReserved;
    }
    static bool qEventRegular(EventId eventId) {
        return ReservedEvent_FirstNonReserved <= eventId
               && eventId <= EventId_LastNonReserved();
    }


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
    bool m_Kelf;
};

}}

#endif // KCC_EVENTS_EVENTMGR_H


#include <set>


#include "arch/inc/pearray.hpp"

#include "events/inc/events.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisadmatrigger.hpp"

#include "wave/inc/sbatomwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"
#include "wavecode/inc/wavecodesbatom.hpp"
//#include "wavecode/inc/wavecodesbatomload.hpp"
//#include "wavecode/inc/wavecodesbatomsave.hpp"

namespace kcc {
namespace wavecode {


//************************************************************************
WaveCodeSbAtom::WaveCodeSbAtom(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



//************************************************************************
void
WaveCodeSbAtom::processOutgoingEdgesAlreadyEmb(wave::SbAtomWaveOp* waveop, events::EventId embEvtId)
{
    const EngineId engineId = waveop->gEngineId();
    bool firstEmb = true;

    std::set<events::EventId> eventIds;
    eventIds.insert(embEvtId);

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        if (firstEmb) {
            firstEmb = false; // this set event is in embedded already in partition N-1
            Assert(succWaveEdge->gEventId() == embEvtId, "Emb event id ", embEvtId,
                    " != first event id ", succWaveEdge->gEventId());
        } else {
            const auto evtId = succWaveEdge->gEventId();
            Assert(eventIds.find(evtId) == eventIds.end(), "Double event id ", evtId);
            eventIds.insert(evtId);

            compisa::SetInstr setInstr;
            AssignWithSizeCheck(setInstr.event_idx, evtId);

            std::ostringstream oss;
            oss << waveop->gOrder() << "-" << waveop->gName();
            m_WaveCode.SaveName(setInstr, oss.str().c_str());
            m_WaveCode.writeInstruction(setInstr, engineId);
        }
    }
} // WaveCodeSbAtom::processOutgoingEdgesAlreadyEmb

//************************************************************************
//======================================================================
kcc_int32
WaveCodeSbAtom::findSuccEventsAndChosenEngine(wave::SbAtomWaveOp* sbAtomWaveop,
                        EngineId& chosenEngId,
                        std::vector<events::EventId>& succEventIds)
{
    chosenEngId = sbAtomWaveop->gEngineId();
    Assert(chosenEngId != EngineId::None, "None engine in waveop ", sbAtomWaveop->gName());
    kcc_int32 numSyncs = 0;
    wave::WaveEdge* chosenPrevEdge = nullptr;

    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (prevWaveEdge->qChosenForSuccSbAtom()) {
            chosenPrevEdge = prevWaveEdge;
            break;
        }
    }
    if (chosenPrevEdge) {
        Assert(chosenPrevEdge->gFromOp()->gEngineId() == chosenEngId,
            "Engine on chosen edge from ", chosenPrevEdge->gFromOp()->gName(), " to ", sbAtomWaveop->gName(),
            " different than engine id ", utils::engineId2Str(chosenEngId));
    }

    // First wait on all other engines
    // Do not use embedded wait because DMA barrier (NOP cycle=128) needs to wait on event too
    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }

        ++numSyncs;
        if (prevWaveEdge->qSyncedWithEvent()) {
            m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, chosenEngId);
        } else if (prevWaveEdge->qSyncedWithSemaphore()) {
            GenerateSemaphoreInstr(prevWaveEdge);
        } else {
            Assert(false, "Must sync edge from ", prevWaveEdge->gFromOp()->gName(),
                   " to ", prevWaveEdge->gToOp()->gName());
        }
    }

    for (auto succWaveEdge : sbAtomWaveop->gSuccWaveEdges()) {
        if (succWaveEdge->qNeedToImplementSync()) {
            if (succWaveEdge->qSyncedWithEvent()) {
                succEventIds.push_back(succWaveEdge->gEventId());
            } else if (succWaveEdge->qSyncedWithSemaphore()) {
                // No need to do anything, DMA will increment semaphore
            } else {
                Assert(false, "Must sync edge from ", succWaveEdge->gFromOp()->gName(),
                       " to ", succWaveEdge->gToOp()->gName());
            }
            ++numSyncs;
        }
    }
    return numSyncs;
}

}}


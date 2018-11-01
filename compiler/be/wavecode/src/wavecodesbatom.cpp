#include <set>


#include "arch/inc/pearray.hpp"

#include "events/inc/events.hpp"
#include "compisa/inc/compisanop.hpp"
#include "compisa/inc/compisadmatrigger.hpp"

#include "wave/inc/sbatomwaveop.hpp"
#include "wavecode/inc/wavecodesbatom.hpp"

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
            setInstr.event_idx  = evtId;

            std::ostringstream oss;
            oss << waveop->gOrder() << "-" << waveop->gName();
            m_WaveCode.SaveName(setInstr, oss.str().c_str());
            m_WaveCode.writeInstruction(setInstr, engineId);
        }
    }
}

//************************************************************************
void
WaveCodeSbAtom::addDmaBarrier(EngineId engId)
{
    const arch::PeArray& peArray(arch::Arch::gArch().gPeArray());
    compisa::NopInstr nopInstr;
    nopInstr.inst_events.wait_event_mode    = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
    nopInstr.inst_events.wait_event_idx     = 0;
    nopInstr.inst_events.set_event_mode     = events::eventSetMode2Isa(events::EventSetMode::DontSet);
    nopInstr.inst_events.set_event_idx      = 0;
    nopInstr.cycle_cnt                      = std::max(peArray.gNumberRows(), peArray.gNumberColumns());
    std::ostringstream oss;
    m_WaveCode.SaveName(nopInstr, "Delay before DMA due to END-WRITE signal being set at end of partition 0");
    m_WaveCode.writeInstruction(nopInstr, engId);
}

//************************************************************************
void
WaveCodeSbAtom::addSecondDmaTrigger(
    compisa::DmaTriggerInstr& dmaTriggerInstr, EngineId chosenEngId)
{
    enum {
        TWO_DMA_TRIGGER_INST = 0,
    };

    if (TWO_DMA_TRIGGER_INST) {
        // dummy
        dmaTriggerInstr.inst_events.wait_event_idx  = 0;
        dmaTriggerInstr.inst_events.wait_event_mode = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
        dmaTriggerInstr.inst_events.set_event_idx   = 0;
        dmaTriggerInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(events::EventSetMode::DontSet);
        m_WaveCode.SaveName(dmaTriggerInstr, "Second DMA_TRIGGER for non-atomic double WRITE");
        m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    }
}

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
    for (auto prevWaveEdge : sbAtomWaveop->gPrevWaveEdges()) {
        if (! prevWaveEdge->qNeedToImplementSync()) {
            continue;
        }
        const wave::WaveOp* const fromWop = prevWaveEdge->gFromOp();

        if (prevWaveEdge == chosenPrevEdge
            && fromWop->gEngineId() == chosenEngId
            && !fromWop->qSbAtomWaveOp())
        {
            continue;
        }

        ++numSyncs;
        m_WaveCode.writeWaitOrWaitClearInstr(prevWaveEdge, chosenEngId);
    }

    for (auto succWaveEdge : sbAtomWaveop->gSuccWaveEdges()) {
        if (succWaveEdge->qNeedToImplementSync()) {
            succEventIds.push_back(succWaveEdge->gEventId());
            ++numSyncs;
        }
    }
    return numSyncs;
}

}}


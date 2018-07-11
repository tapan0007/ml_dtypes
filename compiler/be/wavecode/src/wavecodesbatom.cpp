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
            SaveName(setInstr, oss.str().c_str());
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
        m_WaveCode.writeInstruction(dmaTriggerInstr, chosenEngId);
    }
}

}}


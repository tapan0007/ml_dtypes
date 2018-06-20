#include <set>


#include "events/inc/events.hpp"

#include "wave/inc/sbatomwaveop.hpp"
#include "wavecode/inc/wavecodesbatom.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtom::WaveCodeSbAtom(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



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


}}


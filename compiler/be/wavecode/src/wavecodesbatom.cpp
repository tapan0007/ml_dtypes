
#include "wave/inc/sbatomwaveop.hpp"
#include "wavecode/inc/wavecodesbatom.hpp"

namespace kcc {
namespace wavecode {

WaveCodeSbAtom::WaveCodeSbAtom(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeSbAtom::processOutgoingEdgesAlreadyEmb(wave::SbAtomWaveOp* waveop)
{
    const EngineId engineId = waveop->gEngineId();
    bool firstEmb = true;

    for (auto succWaveEdge : waveop->gSuccWaveEdges()) {
        if (! succWaveEdge->qNeedToImplementWait()) {
            continue;
        }
        if (firstEmb) {
            firstEmb = false; // this set event is in embedded already in partition N-1
        } else {
            SET setInstr;
            setInstr.event_id  = succWaveEdge->gEventId();
            m_WaveCode.writeInstruction(setInstr, engineId);
        }
    }
}


}}


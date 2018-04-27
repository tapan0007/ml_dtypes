

#include "compisa/inc/compisanop.hpp"



#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/resaddwaveop.hpp"
#include "wave/inc/nopwaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodenop.hpp"

namespace kcc {
namespace wavecode {

WaveCodeNop::WaveCodeNop(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeNop::generate(wave::WaveOp* waveOp)
{
    auto nopWaveop = dynamic_cast<wave::NopWaveOp*>(waveOp);
    Assert(nopWaveop, "Codegen expects nop waveop");

    //************************************************************************
    if (qParallelStreams()) {
        if (nopWaveop->gPrevWaveEdges().size() > 1
                || nopWaveop->gSuccWaveEdges().size() > 1)
        {
            processIncomingEdges(nopWaveop);
            processOutgoingEdges(nopWaveop);
        } else {
            compisa::NopInstr nopInstr;
            nopInstr.inst_events.wait_event_mode  = events::eventWaitMode2Isa(events::EventWaitMode::DontWait);
            nopInstr.inst_events.set_event_mode   = events::eventSetMode2Isa(events::EventSetMode::DontSet);
            nopInstr.inst_events.wait_event_idx   = 0;
            nopInstr.inst_events.set_event_idx    = 0;
            int numSyncs = 0;

            if (nopWaveop->gPrevWaveEdges().size() > 0) {
                const wave::WaveEdge* prevWaveEdge = nopWaveop->gPrevWaveEdges()[0];
                if (prevWaveEdge->qNeedToSync()) {
                    nopInstr.inst_events.wait_event_mode  = events::eventWaitMode2Isa(prevWaveEdge->gWaitEventMode());
                    nopInstr.inst_events.wait_event_idx   = prevWaveEdge->gEventId();
                    ++numSyncs;
                }
            }
            if (nopWaveop->gSuccWaveEdges().size() > 0) {
                const wave::WaveEdge* succWaveEdge = nopWaveop->gSuccWaveEdges()[0];
                if (succWaveEdge->qNeedToSync()) {
                    nopInstr.inst_events.set_event_mode  = events::eventSetMode2Isa(succWaveEdge->gSetEventMode());
                    nopInstr.inst_events.set_event_idx   = succWaveEdge->gEventId();
                    ++numSyncs;
                }
            }
            Assert(numSyncs > 0, "NOP waveop ", nopWaveop->gName(), " does not sync at all");
            const EngineId engineId = nopWaveop->gEngineId();
            m_WaveCode.writeInstruction(nopInstr, engineId);
        }
    }
}


}}






#include "compisa/inc/compisanop.hpp"



#include "utils/inc/asserter.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/tensortensorwaveop.hpp"
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
        // NopInstr not ready in Inkling yet. "true" forces processIn/OutEdges.
        if (true ||
            nopWaveop->gPrevWaveEdges().size() > 1
                || nopWaveop->gSuccWaveEdges().size() > 1)
        {
            generateWithWaitSet(nopWaveop);
        } else {
            generateWithNop(nopWaveop);
        }
    }
}






void
WaveCodeNop::generateWithWaitSet(wave::NopWaveOp* nopWaveop)
{
    int numSyncs = 0;
    numSyncs += processIncomingEdges(nopWaveop);
    numSyncs += processOutgoingEdges(nopWaveop);
    Assert(numSyncs > 0, "NOP waveop ", nopWaveop->gName(), " does not sync at all");
}


void
WaveCodeNop::generateWithNop(wave::NopWaveOp* nopWaveop)
{
    compisa::NopInstr nopInstr;
    AssignWithSizeCheck(nopInstr.inst_events.wait_event_mode,
                        events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(nopInstr.inst_events.set_event_mode,
                        events::eventSetMode2Isa(events::EventSetMode::DontSet));
    AssignWithSizeCheck(nopInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(nopInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(nopInstr.cycle_cnt, 1);

    int numSyncs                            = 0;

    if (nopWaveop->gPrevWaveEdges().size() > 0) {
        const wave::WaveEdge* prevWaveEdge = nopWaveop->gPrevWaveEdges()[0];
        if (prevWaveEdge->qNeedToSync()) {
            Assert(prevWaveEdge->qSyncedWithEvent(), "Must use event to syn NOP");
            AssignWithSizeCheck(nopInstr.inst_events.wait_event_mode,
                                events::eventWaitMode2Isa(prevWaveEdge->gWaitEventMode()));
            AssignWithSizeCheck(nopInstr.inst_events.wait_event_idx, prevWaveEdge->gEventId());
            ++numSyncs;
        }
    }
    if (nopWaveop->gSuccWaveEdges().size() > 0) {
        const wave::WaveEdge* succWaveEdge = nopWaveop->gSuccWaveEdges()[0];
        if (succWaveEdge->qNeedToSync()) {
            AssignWithSizeCheck(nopInstr.inst_events.set_event_mode,
                                events::eventSetMode2Isa(succWaveEdge->gSetEventMode()));
            AssignWithSizeCheck(nopInstr.inst_events.set_event_idx,
                                succWaveEdge->gEventId());
            ++numSyncs;
        }
    }
    Assert(numSyncs > 0, "NOP waveop ", nopWaveop->gName(), " does not sync at all");
    const EngineId engineId = nopWaveop->gEngineId();
    std::ostringstream oss;
    oss << nopWaveop->gOrder() << "-" <<  nopWaveop->gName();
    m_WaveCode.SaveName(nopInstr, oss.str().c_str());
    m_WaveCode.writeInstruction(nopInstr, engineId);
}


}}




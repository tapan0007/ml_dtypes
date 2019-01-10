
#include "compisa/inc/compisaregshuffle.hpp"



#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/regshufflewaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
//#include "wave/inc/sbatomloadwaveop.hpp"
//#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderegshuffle.hpp"

namespace kcc {
namespace wavecode {


WaveCodeRegShuffle::WaveCodeRegShuffle(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodeRegShuffle::generate(wave::WaveOp* waveOp)
{
    auto regshuffleWaveop = dynamic_cast<wave::RegShuffleWaveOp*>(waveOp);
    assert(regshuffleWaveop);
    //const arch::Arch& arch(arch::Arch::gArch());
    //const auto& stateBuf(arch.gStateBuffer());

    const EngineId engineId = regshuffleWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    compisa::RegShuffleInstr regshuffleInstr;

    AssignWithSizeCheck(regshuffleInstr.start_reg_id, regshuffleWaveop->gStartReg());
    for (kcc_int32 k = 0; k < regshuffleWaveop->gMaxNumShuffleRegs(); ++k) {
        AssignWithSizeCheck(regshuffleInstr.in_sel[k], regshuffleWaveop->gInSel(k));
    }

    AssignWithSizeCheck(regshuffleInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(regshuffleInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(regshuffleInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(regshuffleInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(regshuffleWaveop, regshuffleInstr.inst_events);
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(regshuffleWaveop, regshuffleInstr);
    }

    if (! instructionWritten) {
        std::ostringstream oss;
        oss << regshuffleWaveop->gOrder() << "-" << regshuffleWaveop->gName();
        m_WaveCode.SaveName(regshuffleInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(regshuffleInstr);
    }
}

}}



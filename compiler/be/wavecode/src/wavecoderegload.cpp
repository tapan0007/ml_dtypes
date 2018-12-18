
#include "compisa/inc/compisaregload.hpp"



#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/regloadwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecoderegload.hpp"

namespace kcc {
namespace wavecode {


WaveCodeRegLoad::WaveCodeRegLoad(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodeRegLoad::generate(wave::WaveOp* waveOp)
{
    auto regloadWaveop = dynamic_cast<wave::RegLoadWaveOp*>(waveOp);
    assert(regloadWaveop);
    const arch::Arch& arch(arch::Arch::gArch());

    const auto& stateBuf(arch.gStateBuffer());

    const EngineId engineId = regloadWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    compisa::RegLoadInstr regloadInstr;

    AssignWithSizeCheck(regloadInstr.dtype, regloadWaveop->gInDtype().gSimTypeId());

    initMemAccess(regloadInstr.src_mem_pattern);

    AssignWithSizeCheck(regloadInstr.src_mem_pattern.start_addr,
        stateBuf.gEntryTpbAddress(
            arch.gNumberPeArrayRows()/2 * regloadWaveop->gSrcStartAtMidPart(),
            regloadWaveop->gSrcSbAddress()));

    AssignWithSizeCheck(regloadInstr.num_active_channels, regloadWaveop->gNumPartitions());
    if (regloadWaveop->qParallelMode()) {
        AssignWithSizeCheck(regloadInstr.serialization_mode, TONGA_ISA_TPB_REG_SERIALIZATION_MODE_PARALLEL);
    } else {
        AssignWithSizeCheck(regloadInstr.serialization_mode, TONGA_ISA_TPB_REG_SERIALIZATION_MODE_SERIAL);
    }
    AssignWithSizeCheck(regloadInstr.src_mem_pattern.step_elem[PatDim_X], regloadWaveop->gSrcXStep());
    AssignWithSizeCheck(regloadInstr.src_mem_pattern.num_elem[PatDim_X], regloadWaveop->gSrcXNum());
    AssignWithSizeCheck(regloadInstr.src_mem_pattern.step_elem[PatDim_Y], regloadWaveop->gSrcYStep());
    AssignWithSizeCheck(regloadInstr.src_mem_pattern.num_elem[PatDim_Y], regloadWaveop->gSrcYNum());
    AssignWithSizeCheck(regloadInstr.src_mem_pattern.step_elem[PatDim_Z], regloadWaveop->gSrcZStep());
    AssignWithSizeCheck(regloadInstr.src_mem_pattern.num_elem[PatDim_Z], regloadWaveop->gSrcZNum());


    AssignWithSizeCheck(regloadInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(regloadInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(regloadInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(regloadInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(regloadWaveop, regloadInstr.inst_events);
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(regloadWaveop, regloadInstr);
    }

    if (! instructionWritten) {
        std::ostringstream oss;
        oss << regloadWaveop->gOrder() << "-" << regloadWaveop->gName();
        m_WaveCode.SaveName(regloadInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(regloadInstr);
    }
}

}}



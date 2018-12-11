
#include "compisa/inc/compisareciprocal.hpp"



#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/reciprocalwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodereciprocal.hpp"

namespace kcc {
namespace wavecode {


WaveCodeReciprocal::WaveCodeReciprocal(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}


void
WaveCodeReciprocal::generate(wave::WaveOp* waveOp)
{
    auto reciprocalWaveop = dynamic_cast<wave::ReciprocalWaveOp*>(waveOp);
    assert(reciprocalWaveop);
    const arch::Arch& arch(arch::Arch::gArch());
    const auto& psumBuf(arch.gPsumBuffer());
    const auto& stateBuf(arch.gStateBuffer());

    const EngineId engineId = reciprocalWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for Pool should be Pooling");

    compisa::ReciprocalInstr reciprocalInstr;

    AssignWithSizeCheck(reciprocalInstr.in_dtype, reciprocalWaveop->gInDtype().gSimTypeId());
    AssignWithSizeCheck(reciprocalInstr.out_dtype, reciprocalWaveop->gOutDtype().gSimTypeId());

    initMemAccess(reciprocalInstr.src_mem_pattern);
    if (reciprocalWaveop->qSrcIsPsum()) {
        AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.start_addr,
            psumBuf.gEntryTpbAddress(
                reciprocalWaveop->gSrcPsumBankId(),
                reciprocalWaveop->gSrcPsumBankOffset(),
                reciprocalWaveop->gInDtype()));
    } else { // State buffer
        AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.start_addr,
            stateBuf.gEntryTpbAddress(
                arch.gNumberPeArrayRows()/2 * reciprocalWaveop->gSrcStartAtMidPart(),
                reciprocalWaveop->gSrcSbAddress()));
    }

    AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.step_elem[PatDim_X], reciprocalWaveop->gSrcXStep());
    AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.num_elem[PatDim_X], reciprocalWaveop->gSrcXNum());
    AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.step_elem[PatDim_Y], reciprocalWaveop->gSrcYStep());
    AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.num_elem[PatDim_Y], reciprocalWaveop->gSrcYNum());
    AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.step_elem[PatDim_Z], reciprocalWaveop->gSrcZStep());
    AssignWithSizeCheck(reciprocalInstr.src_mem_pattern.num_elem[PatDim_Z], reciprocalWaveop->gSrcZNum());

    AssignWithSizeCheck(reciprocalInstr.num_active_channels, reciprocalWaveop->gNumPartitions());

    initMemAccess(reciprocalInstr.dst_mem_pattern);
    // For now DST is always StateBuf
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.start_addr, stateBuf.gEntryTpbAddress(
        arch.gNumberPeArrayRows()/2 * reciprocalWaveop->gDstStartAtMidPart(),
        reciprocalWaveop->gDstSbAddress()));

    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.step_elem[PatDim_X], reciprocalWaveop->gDstXStep());
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.num_elem[PatDim_X], reciprocalWaveop->gDstXNum());
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.step_elem[PatDim_Y], reciprocalWaveop->gDstYStep());
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.num_elem[PatDim_Y], reciprocalWaveop->gDstYNum());
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.step_elem[PatDim_Z], reciprocalWaveop->gDstZStep());
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.num_elem[PatDim_Z], reciprocalWaveop->gDstZNum());
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.step_elem[PatDim_W], 0);
    AssignWithSizeCheck(reciprocalInstr.dst_mem_pattern.num_elem[PatDim_W], 1);

    AssignWithSizeCheck(reciprocalInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(reciprocalInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(reciprocalInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(reciprocalInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(reciprocalWaveop, reciprocalInstr.inst_events);
    } // end incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(reciprocalWaveop, reciprocalInstr);
    }

    if (! instructionWritten) {
        std::ostringstream oss;
        oss << reciprocalWaveop->gOrder() << "-" << reciprocalWaveop->gName();
        m_WaveCode.SaveName(reciprocalInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(reciprocalInstr);
    }
}

}}



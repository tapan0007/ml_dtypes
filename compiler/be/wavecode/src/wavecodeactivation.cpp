
#include "compisa/inc/compisaactivate.hpp"


#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/activationwaveop.hpp"
#include "wave/inc/matmulwaveop.hpp"
#include "wave/inc/poolwaveop.hpp"
#include "wave/inc/sbatomloadwaveop.hpp"
#include "wave/inc/sbatomsavewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodeactivation.hpp"

namespace kcc {
namespace wavecode {


WaveCodeActivation::WaveCodeActivation(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeActivation::generate(wave::WaveOp* waveop)
{
    auto activationWaveop = dynamic_cast<wave::ActivationWaveOp*>(waveop);
    assert(activationWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());
    const EngineId engineId = activationWaveop->gEngineId();
    Assert(EngineId::Activation == engineId, "Engine id for Activation waveop should be Activation");
    kelf::DmaDescription& kelfDma(m_WaveCode.gDmaDescription());

    compisa::ActivateInstr activationInstr;

    kelfDma.addActivationFunc(activationWaveop->gActivationFunc());

    AssignWithSizeCheck(activationInstr.activation_func, activationWaveop->gSimActivationFunc());
    AssignWithSizeCheck(activationInstr.in_dtype, activationWaveop->gInDtype().gSimTypeId());
    AssignWithSizeCheck(activationInstr.bias_dtype, activationWaveop->gBiasDtype().gSimTypeId());
    AssignWithSizeCheck(activationInstr.out_dtype, activationWaveop->gOutDtype().gSimTypeId());


    // TODO: for now Activation reads from 0 elem in bank.
    initMemAccess(activationInstr.src_mem_pattern);
    if (activationWaveop->qSrcIsPsum()) {
        AssignWithSizeCheck(activationInstr.src_mem_pattern.start_addr,
                            psumBuf.gEntryTpbAddress(activationWaveop->gSrcPsumBankId(),
                                                     activationWaveop->gSrcPsumBankOffset(),
                                                     activationWaveop->gInDtype()));
    } else {
        AssignWithSizeCheck(activationInstr.src_mem_pattern.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * activationWaveop->gSrcStartAtMidPart(),
                                                      activationWaveop->gSrcSbAddress()));
    }
    AssignWithSizeCheck(activationInstr.src_mem_pattern.step_elem[PatDim_X], activationWaveop->gSrcXStep());
    AssignWithSizeCheck(activationInstr.src_mem_pattern.num_elem[PatDim_X], activationWaveop->gSrcXNum());
    AssignWithSizeCheck(activationInstr.src_mem_pattern.step_elem[PatDim_Y], activationWaveop->gSrcYStep());
    AssignWithSizeCheck(activationInstr.src_mem_pattern.num_elem[PatDim_Y], activationWaveop->gSrcYNum());
    AssignWithSizeCheck(activationInstr.src_mem_pattern.step_elem[PatDim_Z], activationWaveop->gSrcZStep());
    AssignWithSizeCheck(activationInstr.src_mem_pattern.num_elem[PatDim_Z], activationWaveop->gSrcZNum());


    initMemAccess(activationInstr.dst_mem_pattern);
    if (activationWaveop->qDstIsPsum()) {
        AssignWithSizeCheck(activationInstr.dst_mem_pattern.start_addr,
                            psumBuf.gEntryTpbAddress(activationWaveop->gDstPsumBankId(),
                                                     activationWaveop->gDstPsumBankOffset(),
                                                     activationWaveop->gOutDtype()));
    } else {
        AssignWithSizeCheck(activationInstr.dst_mem_pattern.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * activationWaveop->gDstStartAtMidPart(),
                                                      activationWaveop->gDstSbAddress()));
    }
    AssignWithSizeCheck(activationInstr.dst_mem_pattern.step_elem[PatDim_X], activationWaveop->gDstXStep());
    AssignWithSizeCheck(activationInstr.dst_mem_pattern.num_elem[PatDim_X], activationWaveop->gDstXNum());
    AssignWithSizeCheck(activationInstr.dst_mem_pattern.step_elem[PatDim_Y], activationWaveop->gDstYStep());
    AssignWithSizeCheck(activationInstr.dst_mem_pattern.num_elem[PatDim_Y], activationWaveop->gDstYNum());
    AssignWithSizeCheck(activationInstr.dst_mem_pattern.step_elem[PatDim_Z], activationWaveop->gDstZStep());
    AssignWithSizeCheck(activationInstr.dst_mem_pattern.num_elem[PatDim_Z], activationWaveop->gDstZNum());

    activationInstr.scale_value = activationWaveop->gScale(); // float

    AssignWithSizeCheck(activationInstr.bias_addr,
                        stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * activationWaveop->gBiasStartAtMidPart(),
                                                  activationWaveop->gBiasSbAddress()));

    AssignWithSizeCheck(activationInstr.num_active_channels, activationWaveop->gNumPartitions());

    AssignWithSizeCheck(activationInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(activationInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(activationInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(activationInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(activationWaveop, activationInstr.inst_events);
    } // incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(activationWaveop, activationInstr);
    }



    if (! instructionWritten) {
        std::ostringstream oss;
        oss << activationWaveop->gOrder() << "-" <<  activationWaveop->gName();
        m_WaveCode.SaveName(activationInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(activationInstr);
    }
}


}}



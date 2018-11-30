
#include "compisa/inc/compisatensorscalarop.hpp"


#include "utils/inc/asserter.hpp"
#include "events/inc/events.hpp"

#include "arch/inc/arch.hpp"
#include "arch/inc/psumbuffer.hpp"

#include "wave/inc/waveedge.hpp"
#include "wave/inc/clipbyvaluewaveop.hpp"

#include "wavecode/inc/wavecode.hpp"
#include "wavecode/inc/wavecodeclipbyvalue.hpp"

namespace kcc {
namespace wavecode {


WaveCodeClipByValue::WaveCodeClipByValue(WaveCodeRef waveCode)
    : WaveCodeWaveOp(waveCode)
{}



void
WaveCodeClipByValue::generate(wave::WaveOp* waveop)
{
    auto clipByValueWaveop = dynamic_cast<wave::ClipByValueWaveOp*>(waveop);
    assert(clipByValueWaveop);

    const arch::Arch& arch(arch::Arch::gArch());
    const arch::PsumBuffer& psumBuf(arch.gPsumBuffer());
    const arch::StateBuffer& stateBuf(arch.gStateBuffer());
    const EngineId engineId = clipByValueWaveop->gEngineId();
    Assert(EngineId::Pooling == engineId, "Engine id for ClipByValue waveop should be Pooling engine");

    compisa::TensorScalarOpInstr tensorScalarOpInstr;

    AssignWithSizeCheck(tensorScalarOpInstr.in_dtype, clipByValueWaveop->gInDtype().gSimTypeId());
    AssignWithSizeCheck(tensorScalarOpInstr.out_dtype, clipByValueWaveop->gOutDtype().gSimTypeId());


    // TODO: for now Activation reads from 0 elem in bank.
    initMemAccess(tensorScalarOpInstr.src_mem_pattern);
    if (clipByValueWaveop->qSrcIsPsum()) {
        AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.start_addr,
                            psumBuf.gEntryTpbAddress(clipByValueWaveop->gSrcPsumBankId(),
                                                     clipByValueWaveop->gSrcPsumBankOffset(),
                                                     clipByValueWaveop->gInDtype()));
    } else {
        AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * clipByValueWaveop->gSrcStartAtMidPart(),
                                                      clipByValueWaveop->gSrcSbAddress()));
    }
    AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.step_elem[PatDim_X], clipByValueWaveop->gSrcXStep());
    AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.num_elem[PatDim_X], clipByValueWaveop->gSrcXNum());
    AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.step_elem[PatDim_Y], clipByValueWaveop->gSrcYStep());
    AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.num_elem[PatDim_Y], clipByValueWaveop->gSrcYNum());
    AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.step_elem[PatDim_Z], clipByValueWaveop->gSrcZStep());
    AssignWithSizeCheck(tensorScalarOpInstr.src_mem_pattern.num_elem[PatDim_Z], clipByValueWaveop->gSrcZNum());


    initMemAccess(tensorScalarOpInstr.dst_mem_pattern);
    if (clipByValueWaveop->qDstIsPsum()) {
        AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.start_addr,
                            psumBuf.gEntryTpbAddress(clipByValueWaveop->gDstPsumBankId(),
                                                     clipByValueWaveop->gDstPsumBankOffset(),
                                                     clipByValueWaveop->gOutDtype()));
    } else {
        AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.start_addr,
                            stateBuf.gEntryTpbAddress(arch.gNumberPeArrayRows()/2 * clipByValueWaveop->gDstStartAtMidPart(),
                                                      clipByValueWaveop->gDstSbAddress()));
    }
    AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.step_elem[PatDim_X], clipByValueWaveop->gDstXStep());
    AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.num_elem[PatDim_X], clipByValueWaveop->gDstXNum());
    AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.step_elem[PatDim_Y], clipByValueWaveop->gDstYStep());
    AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.num_elem[PatDim_Y], clipByValueWaveop->gDstYNum());
    AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.step_elem[PatDim_Z], clipByValueWaveop->gDstZStep());
    AssignWithSizeCheck(tensorScalarOpInstr.dst_mem_pattern.num_elem[PatDim_Z], clipByValueWaveop->gDstZNum());

    AssignWithSizeCheck(tensorScalarOpInstr.num_active_channels, clipByValueWaveop->gNumPartitions());

    AssignWithSizeCheck(tensorScalarOpInstr.op[0], TONGA_ISA_TPB_ALU_OP_MIN);
    tensorScalarOpInstr.imm_val_float[0] =  clipByValueWaveop->gMaxValue(); // float

    AssignWithSizeCheck(tensorScalarOpInstr.op[1], TONGA_ISA_TPB_ALU_OP_MAX);
    tensorScalarOpInstr.imm_val_float[1] = clipByValueWaveop->gMinValue(); // float



    AssignWithSizeCheck(tensorScalarOpInstr.inst_events.wait_event_idx, 0);
    AssignWithSizeCheck(tensorScalarOpInstr.inst_events.wait_event_mode, events::eventWaitMode2Isa(events::EventWaitMode::DontWait));
    AssignWithSizeCheck(tensorScalarOpInstr.inst_events.set_event_idx, 0);
    AssignWithSizeCheck(tensorScalarOpInstr.inst_events.set_event_mode, events::eventSetMode2Isa(events::EventSetMode::DontSet));

    //************************************************************************
    if (qParallelStreams()) { // incoming events
        processIncomingEdges(clipByValueWaveop, tensorScalarOpInstr.inst_events);
    } // incoming events


    //************************************************************************
    bool instructionWritten = false;
    if (qParallelStreams()) { // Outgoing events
        instructionWritten = processOutgoingEdges(clipByValueWaveop, tensorScalarOpInstr);
    }


    if (! instructionWritten) {
        std::ostringstream oss;
        oss << clipByValueWaveop->gOrder() << "-" <<  clipByValueWaveop->gName();
        m_WaveCode.SaveName(tensorScalarOpInstr, oss.str().c_str());
        m_WaveCode.writeInstruction(tensorScalarOpInstr);
    }
}


}}


